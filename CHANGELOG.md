## Changelog

Releases follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a minor bump may break API.

### Unreleased

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
larger absolute saving: **1.46x** for iteration mode and **1.32x** for subsample
mode, with peak memory during a subsample run falling from 360 MB to 6 MB at
p=3,000 — that figure is exactly the float64 copy of the expression matrix
described below. Subsample mode gains less by construction: every run draws a
different subset of cells, so its column norms genuinely change and cannot be
hoisted, and it uses the lazy column cache rather than the full matrix.

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
