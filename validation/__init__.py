"""STRIDE validation & benchmarking framework (Phase 2).

This package is **completely separate** from the production ``mechanism`` package:

* it lives outside the packaged ``src/`` tree, so it is never part of the installed
  ``mechanism`` distribution;
* it imports ``mechanism`` only through its documented public API (and, at milestone
  V0, imports nothing from it at all);
* ``mechanism`` never imports this package.

Deleting this directory leaves the production framework (M0–M6) fully functional and
golden-green. See ``VALIDATION_ROADMAP.md`` for the milestone plan (V0–V8).
"""
from ._seed import make_rng, spawn_seeds
from .types import RegionTruth, GroundTruthSystem, SimResult, SweepCell

__version__ = "0.1.0+v0"

__all__ = [
    "make_rng",
    "spawn_seeds",
    "RegionTruth",
    "GroundTruthSystem",
    "SimResult",
    "SweepCell",
    "__version__",
]
