# Design-guided gene selection

:::{admonition} Exploratory
:class: caution
New in 0.7.0, measured on simulated data and on one real cohort (below). The
mechanism and the readout are exact; how it behaves across experimental designs
is not yet known. Expect defaults and names to move.
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

### On a real cohort

Wilk et al. 2020 (PBMCs, seven COVID-19 patients and six healthy donors, 44k
cells after QC, 2,000 HVGs, ten latent dimensions, defaults with 400
iterations). Disease is nested in donor, so no donor covariate was given; the
number to read is the disease AUROC *within* cell types.

| | within-cell-type disease AUROC | cell-type kNN accuracy |
| --- | --- | --- |
| PCA, 10 components | 0.88 | 0.82 |
| scVI, 10 dimensions | 0.95 | 0.88 |
| BAE, plain | 0.80 | 0.79 |
| BAE, `design_key="disease"` | 0.90 (0.88 from dimension 0 alone) | 0.78 |
| shuffled labels | dimension 0 alone: 0.53 | 0.78 |

The design dimension held 24 genes: an interferon block (IFI27, IFI44, IFI44L,
IFI6, IFIT3, MX1, XAF1), the plasmablast expansion (IGHG1, IGHG4, IGLC3, JCHAIN),
the CD16 monocyte and class II changes (FCGR3A, MS4A7, HLA-DQB1), S100A8 and
S100A9, SOCS3 and cytotoxic markers — the paper's headline findings in one
dimension. 96% of its boosting steps were decided by the design term; the
interferon genes entered on both objectives (rank 0 without the design step),
S100A8, GNLY and SOCS3 only because of it (ranks 69, 612 and 390). The other
nine dimensions and the reconstruction quality were unchanged. XIST was selected
too: sex is unbalanced between the cohorts, and the term finds any gene that
tracks the labels, so a sex covariate belongs in `batch_key` on such a cohort.
Donor as `batch_key` removed the disease signal entirely, from the BAE and from
scVI alike, as nesting predicts. With 1,000 iterations, `boosting_nu=0.3`,
`batch_size=1024` and a `(64, 128)` decoder every BAE arm improved (guided 0.92,
dimension 0 alone 0.90, plain 0.82) at about twice the genes per dimension.

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

On the Wilk cohort the stratified form concentrated the design dimension on the
myeloid response: AUROC 0.97 within CD14 monocytes and 0.93 within dendritic
cells and neutrophils, 0.65 to 0.77 within lymphocytes, with FCGR1A, CSF3R,
MS4A7, TMEM176B and SOCS3 replacing the immunoglobulin and cytotoxic genes of
the unstratified dimension. Removing the cell-type main effect drops the
pan-lymphoid interferon component, so the pooled separation is lower (0.86
against 0.90) while the within-monocyte readout is sharper. Which is wanted
depends on the question.

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
plot_selection_paths(adata, dims=[0])      # one line per gene, each jump coloured by what decided it
```

The per-step increments of a dimension sum to its coefficients, so the trace plot
is the coefficient column of the panel unrolled in time. The parts are drawn
side by side rather than stacked because they can have opposite signs.

## Several design variables: one block each

With several variables every variable gets its own block of dimensions. The list
form assigns consecutive blocks in order, each as wide as the variable's encoded
columns; the dict form assigns them explicitly. `design_lambda` may be a dict
with one strength per variable; variables it does not name get `1`.

```python
model.fit(adata, design_key=["condition", "disease"])          # condition -> dims [0, 1], disease -> [2]
model.fit(adata, design_key={"condition": [0, 1], "disease": [4]})
BAEConfig(design_lambda={"condition": 1.0, "disease": 0.5})    # a float still applies to all

adata.uns["bae"]["design_blocks"]                # {variable: dims}
adata.uns["bae"]["design_lambda"]                # {variable: strength}
adata.uns["bae"]["latent_design_r2_per_block"]   # each block's dims scored against its own variable
```

A block keeps the part of the target its variable explains *beyond* the other
design variables, `P_J − P_{J∖v}` with `J` the joint design: the unique part.
For one variable this is the projector of the previous sections. For two
confounded variables it means that what they share is filtered out of *both*
blocks, the conservative choice: a block cannot carry signal the data cannot
tell apart from another variable's. On the Wilk cohort, where every COVID-19
donor is male and two of the six healthy donors are female, sex and disease
share most of their between-group variance, and `design_key={"disease": [0],
"sex": [1]}` gives each block only the part the other cannot explain
(TODO_BLOCKS). `design_dims` remains valid for a single variable; with several,
use the dict.

## Where does a gene's score come from? The decomposition

The design term above changes the fit. The decomposition does not: it reads the
*plain* fit's reconstruction gradient through the design variables and says,
for every boosting step, how much of the selected gene's score came from
residual variance the design explains, and which gene that part alone would
have picked. It measures what a design term would be up against, in the
unguided model, and it works on a guided fit too, where it splits the
reconstruction part.

```python
model.fit(adata, decompose_key="disease")
model.fit(adata, decompose_key=["disease", "sex"], decompose_within="cell_type")

adata.varm["BAE_encoder_weights_between_disease"]   # + "_between_sex", "_shared", "_strata" (or "_mean"), "_within", "_carry"
adata.varm["BAE_residual_variance_share"]           # (n_genes, n_parts): each gene's residual sum of squares, split the same way
adata.uns["bae"]["residual_variance_share_parts"]   # the column names
adata.uns["bae"]["selection_trace"]                 # the same fields as above, keyed by these parts
```

The mathematics is Pythagoras. With `R = D(z) − X` the residual matrix and
`Q_S` an orthonormal basis of the subspace spanned by the encoded columns of a
set of design variables `S` (plus the intercept, or the stratum indicators
under `decompose_within`), the reconstruction loss `L = (1/p)‖R‖²` splits
exactly into `G_S = (1/p)‖Q_SᵀR‖²`, the residual sum of squares the design
explains, and `L − G_S`, the rest. The gradients split the same way, so the
boosting target splits into additive parts and the per-gene selection scores
`xⱼᵀr` add across parts — measured to 6e-8 and 9e-7 relative on the real
decoder. For several variables the split is a commonality analysis:
`between_v = G_J − G_{J∖v}` is what `v` explains beyond the others,
`shared = (G_J − G_∅) − Σ_v between_v` what they explain jointly but no one
uniquely, `mean` (or `strata`) the column means of the residual (the decoder's
bias miss, zero on centred data) or the stratum main effect, and
`within = L − G_J`. One extra backward pass per distinct subset, so for `m`
variables `m + 2` passes.

Three caveats. It decomposes the *residual*, not the data: early in training
the between part is the design-explained variance of the genes, later it is
the design-explained part of what the model still misses. It is exact for the
squared loss only. And `shared` can be negative under suppression and is large
when variables are confounded, which is information about the design, not a
defect of the split.

Read it as before: the parts are additive in the scores, but the pick is the
argmax of their sum, so *which part decided* is the counterfactual column.
`counterfactual_gene["between_disease"]` is the gene the between-disease
residual alone would have selected given the genes already entered; where the
plain fit's pick differs from it, reconstruction went elsewhere, and
`rank["between_disease"]` says how far down that part had the winner.
`BAE_residual_variance_share` is the per-gene version: the share of a gene's
residual variance that sits between the design groups, a design-free score of
every gene, not only the selected ones.

TODO_DECOMP_NUMBERS

```python
plot_selection_trace(adata, dims=[0], decided_by="within")          # hollow marker: within alone would have picked another gene
plot_selection_paths(adata, dims=[0], decided_by="between_disease")
```

`decided_by` names the part to flag against and defaults to `no_design` when a
design term is on, else `within`.

## Cost

The parts ride along as extra target columns in the same `allboost` call, sharing
the covariance cache; the leaders' predictor–target product is computed apart
from the followers' so that the fit stays bitwise the plain one. Measured at
2,000 cells and 2,000 or 5,000 genes over 20 iterations, the fit time with a
`design_key` was within run-to-run noise of the plain fit. The decomposition
adds one backward pass per subset of its variables (TODO_COST). The stored trace
is a few `(latent_dim, stepno)` arrays plus the `(n_genes, n_parts)` shares.

## Not in this release

`design_key` cannot be combined with a transfer model
({meth}`~structboost.BAE.from_reference`), with a warm start (`init_pca`,
`init_obsm`) or with `boosting_independent=False`; each raises with the reason.
The first two would need their own part in the split.

## Related

- {doc}`batch-integration`: the same encoding machinery, with the opposite goal
- {doc}`gene-selection`: whether the gene list is reproducible at all
- {doc}`../concepts/how-it-works`: the loop this hooks into
