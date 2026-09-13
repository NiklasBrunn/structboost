# Design-guided gene selection

:::{admonition} Exploratory
:class: caution
New in 0.7.0 and measured on simulated data only. The mechanism and the
readout are exact; what is not yet known is how it behaves on real experimental
designs. Expect defaults and names to move.
:::

```python
model.fit(adata, design_key="condition")                  # dims default: first min(q, latent_dim)
model.fit(adata, design_key="timepoint", design_dims=[0, 1, 2])
model.fit(adata, design_key=["timepoint", "condition"], batch_key="batch")
```

Reconstruction loss favours the major variance axes. A condition effect that
touches ten genes by one standard deviation is real, and it is also a rounding
error next to cell type, so a plain fit rarely gives it a dimension of its own.
`design_key` names an obs column, or several, and pulls the latent dimensions in
`design_dims` toward it. The other dimensions stay reconstruction-only.

## What the loss is

On the constrained dimensions the fit adds

```
L_design = ½ Σ_k ‖(I − P) z_k‖²
```

summed over cells like the reconstruction target, where `P` projects onto the
encoded design columns plus an intercept. That is the latent variance the design
does **not** explain: for a categorical design, the within-group deviations. Its
gradient is `(I − P) z_k`, so one step removes a fraction `lr · design_lambda` of
the within-group residual from the boosting target and leaves the
design-explained part, the group means, untouched. The step is taken after the
reconstruction step and the orthogonalization, so the constraint is exact on the
constrained dimensions; the orthogonality between constrained and free
dimensions becomes approximate.

Two consequences follow from the loss being quadratic with unit curvature.

**`design_lambda` lives in `[0, 1]`** at the default `target_optim_lr`. At `1`
the target of a constrained dimension is exactly its group-mean pattern. Beyond
`1` the residual comes back with its sign flipped, and since the selection
criterion is squared its competitors regain their scores, so `fit` raises rather
than warns.

**It filters, it does not amplify.** Nothing pushes groups apart; within-group
variation is removed and whatever between-group signal the reconstruction
gradient carries is what remains. The latent scale therefore cannot blow up, and
a dimension whose target holds no design signal is simply left small.

## What to expect from the dial

Selection compares squared scores. Halving the within-group residual only quarters
a competitor's score, and a subtle between-group signal still loses. Measured on
planted data (1,000 cells, 500 genes, three cell types, ten non-marker genes
shifted by one standard deviation between two conditions, three seeds, `latent_dim=6`,
`design_dims=[0]`, 150 iterations):

| `design_lambda` | design R² of dim 0 | condition genes in dim 0 (precision / recall) | marker F1 on the free dims |
| --- | --- | --- | --- |
| 0 | 0.001 | 0 / 0 | 0.63 |
| 0.25 | 0.001 | 0 / 0 | 0.63 |
| 0.5 | 0.001 | 0 / 0 | 0.74 |
| 1 | **0.72** | **1.0 / 1.0** | 0.84 |
| 1, labels shuffled | 0.12 | 0 / 0 | 0.82 |

Below `1` nothing happened on this data. At `1` the constrained dimension held
exactly the ten condition genes, and the free dimensions recovered cell-type
markers *better* than in the plain fit, because the condition signal no longer
competed for them. So sweep `1`, `0.9`, `0.75` downward rather than upward from
small values, and always fit the shuffled-label null alongside: the term will
find *some* genes correlated with any labelling, and on this data that null sits
at R² 0.12 over 29 genes.

:::{warning}
A batch confounded with the design will be selected *as* the design. Pass
`batch_key` too, which keeps the batch columns as mandatory regressors in the
boosting fit, so a gene cannot be selected for a design difference that the batch
explains. Cell type as the design variable mostly re-labels what reconstruction
already separates. A numeric design column gives a linear trend, not groups; pass
timepoints as a categorical column for groups.
:::

## Heterogeneous data: keep the effect inside each cell type

When cell-type composition differs between conditions, a dimension that
separates the conditions can do so by separating cell types. `design_within`
names the strata, typically the annotated cell type, and changes what the
design term keeps: the design effect *within* each stratum. The stratum main
effect is removed together with the within-cell residual, so a constrained
dimension separates the conditions inside every cell type and does not separate
the cell types themselves. This is the stratified formulation of the PerturbBoost
supplement (its Eq. 10), realised with two projectors, one onto the stratum
indicators and one onto the stratum-by-design interaction; the kept part is
their difference.

```python
model.fit(adata, design_key="disease", design_within="cell_type")
adata.uns["bae"]["latent_design_r2_per_dim"]   # share of within-cell-type variance the design explains
```

Disease nested in donor, as in a case-control cohort, is the situation where
this matters most, and it is also the situation where a donor `batch_key` must
*not* be given: every donor is one condition, so mandatory donor regressors in
the boosting fit absorb the disease effect entirely.

`design_dims` defaults to the first `min(q, latent_dim)` dimensions, `q` being the
number of encoded design columns (levels minus one per categorical column). The
design subspace has dimension `q`, so more constrained dimensions than that
cannot all be design-explained and mutually orthogonal. Check the result with

```python
adata.uns["bae"]["latent_design_r2_per_dim"]   # near one on design_dims is the goal
adata.uns["bae"]["latent_obs_r2_per_dim"]      # the batch counterpart, near zero
```

The design term also applies inside {meth}`~structboost.BAE.stability_selection`,
which continues the same alternation.

## The readout: an exact split of every weight

Componentwise boosting is linear in its target once the selection path is fixed:
every coefficient update is `ν · xⱼᵀr / ‖xⱼ‖²`, the residual is linear in the
target, and the mandatory pre-step is a linear solve. So if the target is a sum of
parts, every update, and therefore every final coefficient, is the same sum of
parts. With a `design_key` the fit replays the selection path of the total target
on each part and stores the result for the restored iteration:

```python
adata.varm["BAE_encoder_weights_carry"]    # the code carried over from the previous iteration
adata.varm["BAE_encoder_weights_recon"]    # the reconstruction gradient step
adata.varm["BAE_encoder_weights_design"]   # what the design step removed
# carry + recon + design == BAE_encoder_weights, to the encoder's float32 rounding
```

Under `disentanglement="correlation"` a fourth part, `correlation`, holds that
penalty's step. The split is of *one iteration's* target, not an accumulation
over the fit, and that is deliberate: an accumulated split is exact too but
ill-conditioned — wherever reconstruction keeps re-injecting what the design term
keeps removing, the two accumulated parts grow while cancelling, measured at 400
times the weight on the planted data at `design_lambda=0.5`.

### Read it with the mechanism in mind

On the genes that win, **the design part is a small shrinkage with the opposite
sign of the weight**, and the reconstruction part exceeds the total. That looks
backwards until the mechanism is recalled: the design step removes within-group
variation, so its own additive contribution to a condition gene is minus that
gene's within-group correlation with the target, while the between-group signal
it *protected* sits in the reconstruction gradient. The design term acts by
suppressing the competitors, and a decomposition of the winner's coefficient
cannot show the losers.

What shows the losers is the selection trace:

```python
trace = adata.uns["bae"]["selection_trace"]
trace["gene"]                              # (latent_dim, stepno) gene selected at each step
trace["delta"]["total" | "carry" | "recon" | "design"]   # its increment, split
trace["counterfactual_gene"]["no_design"]  # what the target without the design step would have picked
trace["counterfactual_gene"]["recon"]      # what the reconstruction step alone would have picked
trace["rank"]["no_design"]                 # rank of the applied gene under that target (0 = its own first choice)
trace["fit_score"][part]                   # 1 - relative residual the applied update leaves in that part
trace["alignment"][part]                   # cosine between the applied gene and that part's residual
trace["gain"][part]                        # first-order decrease of that part's residual, <r_part, Δh>
trace["target_norm"][...]                  # per-dimension norm of each target part
```

`fit_score`, `alignment` and `gain` are the candidate-specific scores of the
PerturbBoost supplement (its Eq. 13 to 20), evaluated for the applied update
against the residual of each part, `total`, `no_design`, `recon` and `design`.
They cost nothing extra: `allboost` tracks each residual's norm in O(1) per step.
On the winning genes the `design` alignment is negative, for the reason above.
`track_selection_path=True` additionally logs the selected gene and its increment
at every iteration, as `uns["bae"]["selection_path"]`, the lightweight
selection-stability record; the scores are kept for the restored iteration only.

One choice deliberately not taken from the supplement is its Eq. 9 loss,
`log S_W − α log S_B`. Its per-cell gradient scales as `1/S_W`, so at the latent
scale boosting shrinkage produces it would need a calibration per dataset, and
its between-scatter term adds nothing to *selection* that full filtering does not:
selection compares between-group against within-group scores, and any
amplification of the former is equivalent to a stronger filter of the latter up
to an overall scale, while the amplification would let the latent scale grow.

`counterfactual_gene["no_design"]` is the direct answer to *was this gene selected
because of the design term*: at each step, given the genes already entered and
the adapted penalties, the gene the same boosting run would have taken without
the design step. Where it differs from `gene`, the design term decided the pick.
It is conditional on the shared history, not the trajectory a reconstruction-only
fit would have followed. On the planted data at `design_lambda=1`, 79% of the
constrained dimension's steps were decided this way, and the design part
opposed the weight's sign on every selected condition gene; at `0` the column is
identical to `gene`, as it must be.

`target_norm` carries a null signature worth checking. On real signal the design
part's norm was 0.76 of the total target's; on shuffled labels it was 15 times
the total, because the filter removed nearly the whole target and left a
dimension made of noise. A design part far larger than the total says the
dimension has no design-explained signal to keep.

```python
from structboost import plot_latent_dimensions, plot_selection_trace

plot_latent_dimensions(adata, dims=[0], panels=("scores", "attribution", "groups"), group_by="condition")
plot_selection_trace(adata, dims=[0])      # hollow marker: the design step decided this pick
```

The per-step increments of a dimension sum to its coefficients, so the trace plot
is the coefficient column of the panel unrolled in time. The parts are drawn
side by side rather than stacked because they can have opposite signs.

## Cost

The parts ride along as extra target columns in the same `allboost` call, sharing
the covariance cache and the single predictor–target product. Measured at 2,000
cells and 2,000 or 5,000 genes over 20 iterations, the fit time with a
`design_key` was within run-to-run noise of the plain fit. The stored trace is a
few `(latent_dim, stepno)` arrays.

## Not in this release

`design_key` cannot be combined with a transfer model
({meth}`~structboost.BAE.from_reference`), with a warm start (`init_pca`,
`init_obsm`) or with `boosting_independent=False`; each raises with the reason.
The first two would need their own part in the split.

## Related

- {doc}`batch-integration`: the same encoding machinery, with the opposite goal
- {doc}`gene-selection`: whether the gene list is reproducible at all
- {doc}`../concepts/how-it-works`: the loop this hooks into
