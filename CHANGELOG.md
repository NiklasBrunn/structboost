## Changelog

Releases follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a minor bump may break API.

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
