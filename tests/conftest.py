"""Pytest configuration for scpyutils.

The repo may have a broken editable install that sets ``scpyutils.__path__`` to the
repository root (so ``scpyutils.picutils`` resolves to top-level ``picutils.py``).
Prefer the real package directory ``<repo>/scpyutils/`` for imports.
"""

from pathlib import Path


def pytest_configure(config):
    import sys

    repo = Path(__file__).resolve().parents[1]
    pkg_dir = str(repo / "scpyutils")
    parent = str(repo)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    import scpyutils

    if pkg_dir not in scpyutils.__path__:
        scpyutils.__path__.insert(0, pkg_dir)
