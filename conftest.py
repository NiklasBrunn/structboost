"""Reject a too-old interpreter before anything imports the package.

``pythonpath = ["src", "."]`` in ``pyproject.toml`` makes pytest import this
checkout rather than an installed copy. That is deliberate, so that a bare
``pytest`` tests the code in front of you. It also means nothing was installed,
so nothing enforced ``requires-python = ">=3.10"``. On an older interpreter the
first import instead dies inside ``_boosting.py`` on a runtime ``X | Y`` union,
with a ``TypeError`` that names the two operands and not the cause.

The usual way this happens is a stale ``pytest`` sitting earlier on ``PATH`` than
the project's own interpreter, so the message names that and the way out.

This file has to stay importable on the versions it rejects: no ``X | Y``
annotations, nothing newer than the floor below.
"""

import sys

import pytest

#: Keep in step with ``requires-python`` in ``pyproject.toml``.
MIN_PYTHON = (3, 10)


def pytest_configure(config):
    """Abort with a readable message instead of a TypeError during collection."""
    if sys.version_info < MIN_PYTHON:
        wanted = ".".join(str(part) for part in MIN_PYTHON)
        running = ".".join(str(part) for part in sys.version_info[:3])
        raise pytest.UsageError(
            f"structboost needs Python >= {wanted} but this pytest runs on "
            f"{running} ({sys.executable}). A stale pytest earlier on PATH than "
            "the project interpreter is the usual cause: run 'python -m pytest' "
            "instead, or put the project's environment first on PATH."
        )
