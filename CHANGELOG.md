## Changelog

Releases follow [semantic versioning](https://semver.org). While the project is
pre-1.0, a minor bump may break API.

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
