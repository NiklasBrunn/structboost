"""Keep the documented API and the exported API from drifting apart.

`docs/api/index.rst` is maintained by hand, so a new public symbol is easy to
export and forget to document. It then exists, is importable, and is invisible to
every user reading the docs.

The two checks against that file are skipped when it is absent, which is the case
when the tests are run from an sdist: the sdist ships `tests/` so that downstream
packagers can run them, but not `docs/`. Everything these tests gate is a
repository-consistency property, not a property of the installed package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

API_RST = Path(__file__).resolve().parents[1] / "docs" / "api" / "index.rst"

requires_docs = pytest.mark.skipif(
    not API_RST.is_file(), reason="docs/api/index.rst is not present (running outside a checkout)"
)


def _documented() -> set[str]:
    """Names listed in the autosummary blocks, ignoring prose references."""
    names = re.findall(r"^\s+structboost\.(\w+)\s*$", API_RST.read_text(), flags=re.MULTILINE)
    return {name for name in names if not name.startswith("_")}


@requires_docs
def test_every_exported_symbol_is_documented():
    import structboost

    missing = sorted(set(structboost.__all__) - _documented())
    assert not missing, (
        f"exported but absent from docs/api.rst: {missing}. "
        "Add them there (or drop them from __all__ if they are not public)."
    )


@requires_docs
def test_api_rst_lists_no_symbols_that_do_not_exist():
    import structboost

    extra = sorted(_documented() - set(structboost.__all__))
    assert not extra, f"listed in docs/api.rst but not exported: {extra}"


@pytest.mark.parametrize("name", sorted(__import__("structboost").__all__))
def test_exported_symbol_is_importable(name):
    """`__init__` loads lazily, so a broken export only surfaces on access."""
    import structboost

    assert getattr(structboost, name) is not None
