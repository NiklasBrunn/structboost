## Contributing

Thank you for your interest in contributing to `structboost`.

[Claude Code](https://claude.com/claude-code) (Anthropic) was used in building
this package, to support implementation, to write tests, and to write the
documentation. Individual commits record it as a co-author. You are welcome to
use coding agents on a contribution. What is asked of a pull request is the same
either way: the change is reviewed by you before you open it, the tests pass, and
any behavioural claim added to a docstring is backed by a test or by a
measurement you can point to.

- **Bug reports / feature requests**: please open an issue with a minimal reproduction or clear proposal.
- **Pull requests**:
  - Keep changes focused and add tests where feasible.
  - Run locally:
    - `ruff check .`
    - `ruff format .`
    - `pytest`

### Development install

With [uv](https://docs.astral.sh/uv/), which is much faster and what CI uses:

```bash
uv venv
uv pip install -e ".[bae,plot,dev,test,docs]"
uv run pre-commit install
```

Or with plain pip, which is fully supported:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,test]"
pre-commit install
```

Either way the package builds with Hatchling and end users install it from PyPI
with pip. uv is only a developer convenience. A committed `uv.lock` pins the
CI/development environment for reproducibility. It does not constrain the
dependency ranges that downstream users resolve against. After changing
dependencies in `pyproject.toml`, refresh it with `uv lock` and commit the
result.

### Versioning and releases

Releases follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

While the project is pre-1.0, **a minor bump may break API**. The version
communicates the size of a change, not a compatibility promise. `1.0.0` will
mark the point at which that promise begins.

Every user-facing change (new feature, bug fix, behavioural change) needs:

1. A `CHANGELOG.md` entry under the correct `### [MAJOR.MINOR.PATCH]` heading,
   in the house style, meaning a bold sentence naming the symbol, then *why*, including
   any measurements and rejected alternatives, and a **Migration:** note for
   anything breaking.
2. A matching `version` bump in `pyproject.toml`.

Releases are cut by pushing a tag that matches `pyproject.toml` exactly
(`git tag v0.1.0 && git push --tags`). `.github/workflows/release.yml` verifies
the two agree and refuses to publish if they do not.

