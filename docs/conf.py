from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

project = "structboost"
author = "Niklas Brunn"
copyright = "2026, Niklas Brunn"  # noqa: A001

# Prefer local sources over an installed package when building docs.
sys.path.insert(0, os.path.abspath("../src"))

try:
    release = _pkg_version("structboost")
except PackageNotFoundError:  # pragma: no cover - editable checkout without metadata
    release = ""
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
# `plans/` holds internal design and implementation notes. They are not tracked,
# but they exist in working checkouts, and they are not user documentation — a
# local build should produce the same site CI publishes.
exclude_patterns: list[str] = ["_build", "plans"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# `BoolArray` resolves to `NDArray[np.bool_]`, and the trailing underscore in
# `np.bool_` is read as reference syntax by docutils, which then reports a broken
# target. Mapping the alias keeps the rendered type both correct and quiet.
autodoc_type_aliases = {"BoolArray": "numpy.ndarray"}

# Docstrings are numpy-style throughout; leaving the Google parser on makes
# Napoleon guess, and it guesses wrong on the `Returns` blocks that name no type.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# This package writes `Returns` sections as prose ("Latent representation of
# shape (n_cells, latent_dim).") rather than as a numpydoc `type` line followed
# by an indented description. With `napoleon_use_rtype` on, Napoleon hands that
# sentence to the Python domain as a *type expression*, which tokenises it and
# tries to cross-reference "shape", "n_cells" and even ".". Turning it off puts
# the sentence in the `:returns:` field where it belongs; the actual return type
# still comes from the annotation via `autodoc_typehints`.
napoleon_use_rtype = False

# `autodoc_typehints = "description"` renders every annotation into the parameter
# list, so without these inventories the most common types in this package --
# AnnData (61 uses), np.ndarray (60), torch.Tensor (47) -- are dead text on every
# page.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Surface unresolved cross-references instead of rendering them as plain text.
nitpicky = True
nitpick_ignore_regex = [
    # Internal type aliases and sentinels with no documented target of their own.
    ("py:class", r"structboost\._types\.(ArrayLike|Device)"),
    ("py:class", r"(structboost\._model\.)?_FitLayer"),
    ("py:class", r"Device"),
    # Shorthand dtype spellings used inside hand-written `Parameters` type lines.
    # These are prose, not resolvable targets; numpy's inventory documents the
    # scalar types under different names and `NDArray[...]` not at all.
    ("py:class", r"n(umpy|p)\.\w+"),
    ("py:class", r"numpy\.typing\..*"),
    ("py:class", r"npt?\.NDArray.*"),
    ("py:class", r"NDArray.*"),
    ("py:class", r"(sp|scipy\.sparse)\.spmatrix"),
    ("py:class", r"pd\.DataFrame"),
    ("py:class", r"_CovarianceCache"),
    # Prose that sits in a numpydoc *type* position: "..., optional",
    # "default=False", and shape descriptions such as "(n_cells, n_genes)".
    ("py:class", r"optional|default[ =].*"),
    ("py:class", r"shape|ndarrays?|Path|(ad\.)?AnnData|BoolArray"),
    ("py:class", r"n_(cells|genes|features|samples|targets|leaves|dummy_cols|mandatory)"),
    ("py:obj", r"structboost\..*\.(__init__|training)"),
]

html_theme = "furo"
html_title = f"structboost {release}".strip()

# PyTorch's documentation builds its anchors client-side, so the linkcheck
# builder cannot see them and reports working pages as broken.
linkcheck_anchors_ignore_for_url = [r"https://docs\.pytorch\.org/.*"]
# DOIs are permanent identifiers; resolving to the publisher is what they do.
linkcheck_allowed_redirects = {r"https://doi\.org/.*": r"https://.*"}

myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3
