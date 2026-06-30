"""Path setup for the validation test suite (runs outside production's testpaths).

Makes both packages importable without installing anything:
``validation`` from the repository root, ``mechanism`` from ``src/``.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
