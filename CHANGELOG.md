## Changelog

Releases follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a minor bump may break API.

### [0.6.0] - 2026-09-09

**Not breaking.** Nothing existing changes behaviour; this adds a way to read a
fitted model.

**New `plot_latent_dimensions`: one row per latent dimension, up to three
panels.** The sorted score curve, the per-gene contribution distributions, and
the scores split by a grouping you name. It exists because a fitted BAE has been
readable in principle — the encoder *is* the gene list — and awkward in practice:
`extract_gene_rankings` gives ranked lists, `export_interactive_html` gives an
explorer, and neither says whether a dimension describes a subgroup or a
gradient, nor which cells sit at which end.

**New `plot_dimension_gene_umaps`: a UMAP grid, dimensions by top genes**, drawn
through `scanpy.pl.umap` so the panels match a scanpy figure. It answers the one
question the other panels structurally cannot — whether a dimension's genes light
up the *same* cells — because both the score curve and the contribution violins
have already summed over cells. On real data it separates a coherent programme
from a dimension summing unrelated signals.

**A `"shares"` panel and a new `plot_dimension_correlation`.** The contribution
panel shows a dimension's top handful of genes; the shares panel plots *every*
selected gene's share, sorted, which is the only view that says whether a
dimension rests on three genes or spreads evenly over forty — both legitimate,
and read differently. Ordering it by `rank_by="weight"` while the axis stays on
the share draws the disagreement between the two rankings directly.

The correlation heatmap answers a question no per-dimension panel can: whether
two dimensions are near-duplicates, and therefore whether they can be read as two
findings or one. Pearson by default, Spearman on tie-averaged ranks as an option —
ties are not incidental here, since a sparse encoder leaves many cells at exactly
one value on a subgroup dimension. Signed values get the diverging map centred on
zero; `absolute=True` plots `|r|` with a sequential one, since a magnitude has no
midpoint.

**Genes are ranked by variance share, not `|weight|`,** exposed as the new
`gene_variance_shares`. From `Var(s) = Cov(s, s)` with `s = X @ w` it follows
that `w_g Cov(X_g, s) / Var(s)` decomposes a dimension's variance exactly and
sums to 1, needing no orthogonality assumption and splitting credit correctly
between correlated genes. The weight alone ignores how much a gene actually
varies: measured on 59k PBMCs the two rankings agree on 8.6 of 10 genes per
dimension, and where they differ `|w|` promotes genes that barely move — SCT
ranked #4 by weight and #35 of 38 by share, being near-absent in that tissue.
The share is a **magnitude**: because a negative-weight gene is anti-correlated
with the score, the product is positive either way and only ~1% of genes come out
negative. Direction is therefore the weight's sign, carried by the colour of the
gene name, and the share prints without one.

**The score panel paints in a seeded random order.** Markers are wider than the
spacing between adjacent ranks — about 112 cells overlap at a given x on 59k —
so whatever is drawn last wins every overlap. Painting in score order hands each
contest to the higher-scoring cell, and since score correlates with group, one
group systematically covers its neighbours. Every point keeps its own
`(rank, score)`, so the curve is unchanged and only the paint order is permuted.
This is the *opposite* choice from a UMAP panel, where rare groups are drawn last
on purpose: there position is data and overlap is unavoidable, here every cell
has its own x and any deterministic order is a bias.

**Seven hues, then grey, and the palette is measured rather than chosen** — see
the new `palette_audit`. The package's existing Okabe-Ito set separates colours
from each other under colour-blindness but does not hold contrast against white
at small mark sizes: its `#F0E442` measures 1.32 WCAG contrast against a floor of
3.0, so a cell type holding 5.9% of cells was invisible as 8pt dots while its
violin read perfectly well. **Contrast requirements scale with mark size, so a
palette validated on one mark is not validated on another.** At twelve hues
nothing separates anyway — 1.5 ΔE under simulated deuteranopia against a target
of 8; Kelly's "20 colours of maximum contrast" reaches 0.4, scanpy's `default_20`
0.5. Twenty reliably distinguishable hues do not exist. The default therefore
follows the group count (`tab10`'s first five, then `tab10`, then scanpy's
`default_20`), groups past the palette are grey, and they lose nothing because
the violin panel labels every row — the score panel, which has no labels, is
where the cap binds.

**Signed quantities get a diverging map centred on zero; sequential is for
magnitudes.** The score always, and the gene panels as soon as `scale="zscore"`,
where a sequential map would put its neutral colour at the data mean rather than
at zero. The encoder is bias-free, so zero is a real level — no selected gene
departs from its mean in that cell — and the point at which a dimension's sign
flips.

`scanpy` becomes a new optional extra, needed only by `plot_dimension_gene_umaps`
and imported inside it. Absent, that one function raises naming
`plot_latent_dimensions` as the matplotlib-only alternative; everything else is
unaffected.

Two things are deliberately *not* offered. `c_ig / score_ik`, the intuitive
"fraction of this cell's score", is unusable: the flat plateau of a subgroup
dimension is exactly the cells whose score is ≈0, so the ratio diverges (measured
−8,921 to +1,191 for one gene), flips sign with the score, and inflates without
bound wherever contributions cancel — `normalize="cell"` divides by total
*absolute* contribution instead, which is bounded. And nothing here is
inferential: every number is an exact algebraic decomposition or a descriptive
statistic.

### [0.5.0] - 2026-08-21

**Breaking**, in two parts: the decoder update became one pass over the cells,
and the disentanglement default changed.

**`BAEConfig.decoder_updates_per_iteration` is gone.** Each training
iteration now gives the decoder **one shuffled pass over the cells** —
`ceil(n_cells / batch_size)` AdamW steps, with every cell contributing to exactly
one of them — so the step count is derived from the data rather than set.

The reason is an asymmetry that grew with dataset size. The boosting half of the
alternation is full-batch at every size: `z*` is computed on all cells and the
encoder is re-solved from zero against all cells, every iteration. A fixed
decoder step count made a *cell's* participation depend on how many other cells
existed — at the old `10 × 512` the decoder saw every cell below 5,120 cells,
31% at 16,000 and 5% at 100,000, while the decoder that defines the target for
all of them had been trained on that shrinking slice. Tying the budget to the
data makes both halves consume the same cells per iteration, and it is the same
per-cell invariance that `target_optim_lr` already provides on the target side.

**No quality claim is attached to this change, because none could be
established.** On human pancreas (16,382 cells, 2,000 batch-aware HVGs, nine
protocols, `nu=0.3`, 1000 iterations, one seed) the epoch rule raised in-sample
variance explained from 0.3100 to 0.3217 at `stepno=50` and 0.3414 to 0.3556 at
`stepno=100` — but in-sample reconstruction rewards the arm taking 3.2x the
steps, so that is not evidence. Cell-type silhouette moved the other way,
0.298 → 0.172 and 0.327 → 0.169; on an 80/20 split of the same data it moved the
*opposite* way again (0.265 → 0.333, 0.218 → 0.268). Two conditions, opposite
orderings, one seed each: within run-to-run variance for a method whose encoder
support random-walks. The case for the change is the invariance above, not a
measured improvement.

What is established is the cost: **roughly 2x wall clock** at this dataset size
(99s → 201s at `stepno=50`, 131s → 238s at `stepno=100`), and it grows with cell
count because `max_iterations` does not decay to compensate. scVI, which is
epoch-based in the same way, pairs epochs with a `max_epochs` cap that decays as
1/n; no such heuristic is added here, so raising `batch_size` is the lever.

Keeping a dial (`decoder_epochs_per_iteration = 1.0`, same default, still
tunable) was considered and rejected: the goal was to remove a setting, and a
float multiplier on a pass is a second way to say what `batch_size` already says.
The consequence is that decoder effort per boosting iteration is no longer
adjustable except through `batch_size`.

**Small datasets need a smaller `batch_size` than the default suggests.** Below
about 1,000 cells the default 512 yields one or two steps per iteration, and
effects that require the decoder to move *within* an iteration weaken or reverse.
Measured on 300 cells, a PCA warm start raised the first iteration's loss at one
step per iteration (1.083 against 1.023 for a zero start), tied at ten, and only
paid off at thirty-eight (0.920 against 1.000). The guide documents this.

**Migration:** drop the argument. `BAEConfig(decoder_updates_per_iteration=10)`
now raises `TypeError`. Setting `batch_size = ceil(n_cells / 10)` reproduces the
old *step count*, though not the old batch size, so results will differ either
way.

**`disentanglement="leave_one_out"` is renamed to `"orthogonal"`, and is now the
default.** It replaces the boosting targets with the nearest mutually orthogonal
set of the same column norms — symmetric Löwdin orthogonalization, computed from
the thin SVD as `U @ Vt`, since `T (T'T)^{-1/2} = U V'`. The old name is refused
with a message naming the new one, rather than falling through to the generic
"must be one of" error.

The previous implementation was not an orthogonalization. It regressed each
target column on all the others and kept the residual, which yields
`corr(r_j, r_k) = -rho_{jk|rest}` exactly: the marginal correlation structure
replaced by the *negated partial* correlation structure rather than removed. In
two dimensions that reduces to `corr -> -corr`, an exact sign flip achieving
nothing; in higher dimensions it could amplify a correlation, or manufacture one
between columns that were near-independent. Measured on human pancreas it reached
mean `|corr|` 0.048 against 0.033 for a correct orthogonalization, while selecting
fewer genes (276 against 294) and reconstructing slightly worse.

Symmetric rather than sequential (Gram-Schmidt), which also orthogonalizes but
depends on the order the columns are visited — it returns the last dimension
untouched and strips the first hardest. Latent indices permute freely, which is
why `stability_selection` matches them before counting, so an order-dependent
transform would impose an arbitrary hierarchy. The two measured within seed noise
of each other on both real datasets. Löwdin is also cheaper: `O(n * d**2)` against
`O(n * d**3)` for anything running one least-squares solve per column, measured
6-16x faster and widening with latent width, and it needs no inverse so a
rank-deficient target matrix orthogonalizes instead of raising.

**New `BAEConfig.disentanglement_alpha`, default `1.0`.** Targets become
`(1 - alpha) * targets + alpha * orthogonalized`, so decorrelation can be softened
by choosing an alpha strictly between 0 and 1; `0.0` is equivalent to
`disentanglement="none"`. Interpolation is meaningful because Löwdin returns the
*closest* orthogonal matrix, so the endpoints are already sign-aligned.

**`disentanglement_lambda` default raised from `1e-4` to `1e-2`.** The old default
was inert: on simulated data, Tasic mouse cortex and human pancreas it moved the
mean absolute latent correlation by less than its own seed-to-seed noise, and so
did `1e-3`. On pancreas, mean `|corr|` ran 0.103 (off), 0.100 (`1e-4`), 0.112
(`1e-3`), 0.080 (`1e-2`), 0.061 (`1e-1`). The documented sweep grid, which topped
out at `1e-3`, ended below where the penalty begins to act; the guide now sweeps
upward from `1e-2`.

**Expect the default to cost some biological structure.** Decorrelation is an
extra constraint and real gene programs are not orthogonal, so it trades fidelity
to that structure, and the interpretability resting on it, for a less redundant
representation. Measured against ground truth on simulated data, marker-recovery
F1 moved from 0.98 to 0.88; on Tasic and pancreas the share of latent variance
explained by the annotated cell type fell. `disentanglement="none"` remains a
reasonable choice, and `disentanglement_alpha` exists so the trade can be made
partially.

**Migration:** `disentanglement="leave_one_out"` becomes `"orthogonal"`, or
`"none"` to restore the 0.4.0 default of no constraint. Note that `"orthogonal"`
is not a renamed version of the old behaviour — the method itself changed, so a
0.4.0 fit is reproduced by `"none"`, not by the new name.

**New `BAE.stability_selection(continue_optimizer=...)`, default `False`.** Carries
the decoder's AdamW moment estimates over from `fit` instead of restarting them at
zero. A fresh AdamW restarts its step counter, so bias correction begins again and
`exp_avg_sq` needs on the order of `1/(1 - beta2) = 1000` steps to become a usable
variance estimate; at `ceil(n_cells / batch_size)` steps per iteration that
transient spans roughly `1000 * batch_size / n_cells` of the counted iterations —
brief on a large dataset, most of the window on a small one, and it falls at the
end where the support is furthest from equilibrium.

The state is the one from the iteration `fit` *restored*, not from its last, so the
moments belong to the decoder actually returned. It is held in memory only: `save`
excludes optimizer state, so a model read back from a checkpoint warns and falls
back rather than silently measuring something different. A caller who lowered
`config.decoder_lr` for the counting phase keeps that change — only the moments are
carried across. Default `False` preserves the behaviour every earlier result was
measured under.

**Checkpoints written by 0.4.0 no longer load.** `restore_payload` splats the
stored config into `BAEConfig`, so a format-5 file's `decoder_updates_per_iteration`
is an unexpected keyword argument. The format is bumped to 6 and the loader
refuses 5 by name. No migration is written: the setting no longer exists, so
there is nothing to migrate it to, and a checkpoint is cheap to regenerate.

### [0.4.0] - 2026-08-12

Three defaults change. No configuration fields are added or removed, and anyone
who sets these explicitly is unaffected.

**`enable_early_stopping` now defaults to `False`.** The criterion is a
convergence check being used as a quality check. There is no validation split, so
the training loss cannot see a model that is starting to memorize, and patience
fires long before the model is done: measured against ground truth it stopped at
iteration 90 on a dataset peaking at 259, and at 196 on a simulated scenario
peaking at 560, returning marker-recovery F1 0.502 against 0.787. Across three
real datasets (mouse cortex, human pancreas, human immune) the best iteration
ranged from 154 to 1975 — always past where patience fires.

**No stopping rule replaced it, deliberately.** Latent stability, encoder-support
overlap and held-out reconstruction were each measured as candidates and each
rejected. The representation settles long before gene selection does — on one
dataset consecutive latent codes were rank-identical while the gene set still
turned over 65% cumulatively — and no observable signal tracks the quality peak.
A latent-stability rule was built and tuned; it fired at iteration ~109 on all
three real datasets regardless of where quality peaked, and on one of them it was
worse than not stopping at all. It is not shipped, not even off by default: an
option that should never be enabled is pure carrying cost, which is the same
argument that removed four fields in 0.3.0. `max_iterations=1000` is a defensible
middle of the measured range, not an optimum.

**`BAE.stability_selection(threshold=...)` now defaults to `0.5`, from `0.7`.**
0.7 is too aggressive whenever the latent representation is still moving: across
three real datasets it removed 21-52% of recovered marker genes relative to the
fitted encoder, and 0-7% even when the representation had settled. At 0.5 the
worst loss over the same six runs was 6%. The standalone
`structboost.stability_selection` keeps 0.7, because its Meinshausen-Buhlmann
bound is undefined at or below 0.5.

**The `dim_match_quality` warning now fires below 0.85, from 0.5.** It is the gate
on whether iteration frequencies mean anything: they describe gene-set drift only
if the counted iterations describe one representation. Runs sitting at 0.70-0.79 —
comfortably above the old warning — already lost a quarter to a half of their
recovered markers at the default threshold, while runs at 0.93 and above lost
none. The guide now documents it as a lookup: at least 0.93, either threshold is
safe; 0.70-0.79, use 0.3-0.5 and prefer the flat union.

Also fixed: the gene-selection guide still showed
`fit(adata, stability_selection="iteration")`, which stopped being valid in 0.3.0
when that argument became a bool.

### [0.3.0] - 2026-08-11

**Breaking.** Five settings are gone and `BAE.stability_selection` has one mode
instead of two. Every removed option was off by default, so a fit that took the
defaults is unaffected: the encoder matrix is bitwise identical across a plain
fit, a batch-integrated fit, and both disentanglement methods. What changes is the
surface you have to reason about.

`boosting_nu` stays at `0.1`. Raising it to `0.3` was measured and deferred: at
the default `stepno=50` it matches `0.1`'s marker recovery three to four times
faster and wins outright on low-signal data, but then degrades if training
continues, and the training-MSE stopping rule cannot see it happening. The guide
records the numbers; the two changes belong together and will land together.

**Checkpoints written by 0.2.0 no longer load.** `restore_payload` splats the
stored config into `BAEConfig`, so a dropped field is an unexpected keyword
argument rather than a missing one. The format is bumped to 5 and the loader
refuses 4 by name. No migration is written: the method is under active
development and a checkpoint is cheap to regenerate, whereas a compatibility
shim for options that no longer exist is not.

Removed from `BAEConfig`:

- `decoder_dropout_rate` and `decoder_use_batch_norm`. Batch norm was already
  documented as harmful (marker-recovery F1 0.59-0.73) because the boosting
  target is computed with the decoder in eval mode and the update applied in
  train mode. Dropout has exactly the same inconsistency and it was never written
  down: the target comes from the full network, the update from a thinned one.
  With both gone the decoder is a deterministic per-cell function, which is the
  property `_compute_boosting_targets` has always relied on.
- `disentanglement_standardize`. Both `disentanglement` methods stay.
- `standardize_targets`. It was a genuine trade-off — higher selection precision,
  roughly half the recall — but it is superseded by
  `stability_selection(threshold=...)` -> `stable_encoder()` -> `apply_encoder()`,
  which is the same trade with a dial that reports what it is doing. Two knobs for
  one trade-off is worse than one.

Removed from `fit`:

- `balance_obs`, and with it the `sample_weights` parameter that threaded through
  nine methods and both stability paths. Its own documentation conceded the
  limit: the weights never reached the `allboost` fit, so gene selection stayed
  unbalanced no matter what they were set to. The measured effect was modest (per
  group reconstruction-MSE spread 0.231 -> 0.150) for a mechanism that promised
  more than it delivered.

`BAE.stability_selection` is now iteration mode only; `mode`, `subsample_frac`,
`n_subsamples` and `n_iterations` are gone, and `fit(stability_selection=...)`
takes a bool. Iteration mode was already the default and measures the lower
false-discovery rate (0.26 against 0.31). The subsample path stays available as
the standalone `structboost.stability_selection`, which is where it belongs: it
resamples cells in the Meinshausen-Buhlmann scheme and works on an `allboost`
problem, so supervised users with no training loop to iterate over still have it.
The Meinshausen-Buhlmann bound it reports was measured to be violated by roughly
an order of magnitude when the targets come from a model fitted on the same
cells, and that warning moved with it.

`uns["bae"]["variance_explained"]` is now written by every fit. It used to appear
only when a covariate argument was passed, which left the quality workflow the
guide documents — compare it against `linear_ceiling` — raising `KeyError` on a
plain `fit(adata)`. The metric has nothing to do with covariates; only the
per-group breakdown does, and that stays behind the covariate guard.

Two fixes found along the way: the `allboost` example in the guide passed
`mode="standard"`, an argument `allboost` has never accepted, so it raised
`TypeError` as written. And `disentangle_boosting_targets` projects through the
origin, with no intercept — exact residualization only on centered targets. That
was masked by `disentanglement_standardize`, which centered them; with the flag
gone the assumption is documented instead.

### [0.2.0] - 2026-08-05

The boosting loop got faster without changing what it computes. Measured
end-to-end on real preprocessed scRNA-seq (20,000 cells, `latent_dim=10`):
**1.86x at p=2,000, 2.20x at p=3,000, 1.80x at p=8,000**, and 1.46x on a
batch-integrated fit with 20 covariate dummies.

**This is not a methodological improvement.** Across eighteen configurations —
plain, all three `batch_integration_mode` values, flat and per-dimension
`mandatory_genes`, `balance_obs`, `split_softmax`, both disentanglement methods,
`standardize_targets`, frozen/anchored/zero-dimension transfers and both
stability modes — the selected gene set is *identical* and the reconstruction
loss agrees to the sixth decimal. Nothing here recovers a marker that was
previously missed. What it buys is more iterations and more stability runs for
the same budget, which is what a method whose encoder support keeps drifting
actually needs.

Stability selection benefits too, and for the default `n_runs=300` that is the
larger absolute saving: **1.79–1.88x** for iteration mode and **1.31–1.39x** for
subsample mode at p=2,000–3,000, with peak memory during a subsample run falling
from 360 MB to 5 MB at p=3,000 — that figure is exactly the float64 copy of the
expression matrix described below. Subsample mode gains less by construction:
every run draws a different subset of cells, so its column norms genuinely change
and cannot be hoisted, and it uses the lazy column cache rather than the full
matrix.

Precomputing the covariance matrix *inside* subsample mode was measured and
rejected. It runs 3.07x faster at p=2,000 and 2.69x at p=3,000, but 0.73–0.85x
**slower** from p=6,000 upwards: the number of distinct columns a subsample
selects stays roughly flat as p grows, so the full p x p product stops paying for
itself. A memory guard cannot separate those cases — the 800 MB matrix at
p=10,000 fits comfortably and still loses — so that path stays lazy at every p.

`boosting_precompute_covcache` now defaults to `"auto"` and applies to every
boosting path, including iteration-mode stability selection, which previously
ignored the setting entirely. `"auto"` builds the full p x p covariance matrix
whenever it fits a conservative share of system memory; `True` and `False` are
still honoured exactly, and the resolved decision is recorded in
`adata.uns["bae"]["boosting_precompute_covcache"]`.

The flag was documented as a memory-versus-recomputation trade, which undersold
it. Building all `p` columns at once is one compute-bound matrix product near
hardware peak; fetching them one at a time is a sequence of memory-bound
matrix-vector products. Precomputing wins from roughly `p/59` distinct selected
features onwards — a threshold a fit passes in its first iteration.

Three quantities that never change were being recomputed. `col_norms_sq` was
rebuilt on every training iteration although the design matrix is fixed for the
whole fit, allocating a full-size temporary each time; the mandatory-covariate
block was rebuilt on every *boosting step* although it depends only on the
target; and the covariance cache's NaN check re-scanned columns that had already
been verified. The starting residual correlations are now formed for all latent
dimensions in one matrix product rather than one per dimension.

A float64 target passed against float32 predictors used to make NumPy promote
the *design matrix*, materializing a full float64 copy — measured 2.1x slower,
and it silently produced a different answer. The target is now aligned to the
predictors instead.

The three entry points to the same boosting problem ran at two different
precisions: `fit` in float32, both stability paths in float64 on top of a
float32 copy they had already made. They now agree on float32, which removes two
full-size copies of the expression matrix. The precision this gives up was
measured at about one gene in 380 — far inside the run-to-run support variation
this method documents for itself.

Results are **not bit-identical to 0.1.2**. Coefficients move by ~1e-6 relative
and the selected support does not change; a fixed seed no longer reproduces
0.1.2 output exactly.

### [0.1.2] - 2026-08-03

Project metadata gains `Documentation` and `Changelog` links. PyPI renders
`project.urls` as the sidebar next to the project description, and it carried
only the repository and the issue tracker — so the documentation site, where
every substantive explanation lives, was reachable from the README body but not
from the navigation beside it. Both targets are verified live.

As with 0.1.1, nothing in the package changed: `project.urls` reaches users only
through an upload, so correcting it in the repository has no effect until a
release carries it.

### [0.1.1] - 2026-08-03

A metadata release. No code in `structboost` changed; every difference is in
what the package says about itself.

The PyPI page is the reason for it. A project's long description is baked into
the uploaded artifacts and is immutable per release, so 0.1.0 shipped with a
README announcing "Not on PyPI yet" and directing readers to a pinned TestPyPI
install. Correcting the file in git does not touch the published page — only a
new release does. A `0.1.0.post1` would have expressed "packaging only" more
precisely, but post-releases are handled inconsistently by downstream tooling
and the versioning policy here is plain `MAJOR.MINOR.PATCH`.

Python 3.13 is tested and advertised. `requires-python = ">=3.10"` never had an
upper bound, so pip already installed on 3.13 while the CI matrix stopped at
3.12 — support permitted but never exercised. The matrix now covers it, and the
classifier list says what the requirement already allowed. 3.14 is left out
until the `[bae]` extra's wheels are dependably available there.

`CITATION.cff` gains `version` and `date-released`, which a citation file for a
released version cannot do without, plus a `url` for the documentation site.

### [0.1.0] - 2026-07-31

First public release.

Batch integration is one argument. `BAE.fit(batch_key=...)` names the covariate,
following scVI's spelling, and `batch_integration_mode` chooses between
`"decoder"`, `"encoder"` and `"both"`, defaulting to `"both"`. No `batch_key`
means no integration, and naming a mode without one raises rather than quietly
integrating nothing.

`"encoder"` names the half of the model the mechanism protects, not a tensor the
covariate is fed to: it enters the boosting design as a mandatory regressor so
gene selection is not confounded by it. `transform` remains gene-only and needs
no covariate labels under any mode.

The ridge that stabilizes near-collinear covariates is `BAEConfig.nuisance_ridge`.
It is a numerical knob rather than a modelling one, so it sits with the other
algorithm settings.

The `test` extra pulls the runtime dependencies. The sdist carries `tests/` and
`conftest.py` so that downstream packagers can run the suite at build time, and
with pytest alone that did not work: 360 of the 411 test functions sit behind an
`importorskip` for torch or anndata, so `pip install .[test] && pytest` ran ~51
tests, skipped the rest and reported success. Installing `[test]` now brings in
`[bae]`, so a green build means the suite actually ran. Documenting the
requirement in `CONTRIBUTING.md` instead was rejected, because the reader who
needs it is an automated build script rather than a person.
