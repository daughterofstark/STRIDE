"""STRIDE validation & benchmarking framework (Phase 2).

This package is **completely separate** from the production ``mechanism`` package:

* it lives outside the packaged ``src/`` tree, so it is never part of the installed
  ``mechanism`` distribution;
* it imports ``mechanism`` only through its documented public API, and only from a
  single bridge module (``validation.adapters``); every other non-test module
  (notably ``validation.generate``) is ``mechanism``-free;
* ``mechanism`` never imports this package.

Deleting this directory leaves the production framework (M0-M6) fully functional and
golden-green. See ``VALIDATION_ROADMAP.md`` for the milestone plan (V0-V8).

Milestone V1 adds the Tier-A field-level ground-truth generator and the single
public-API adapter. Note the ``__init__`` intentionally does **not** import
``validation.adapters`` at package import time (that would pull ``mechanism`` into
every ``import validation``); the adapter is imported explicitly where needed. This
keeps ``import validation`` production-free, as the V0 separation guarantee requires.
"""
from ._seed import make_rng, spawn_seeds
from .types import RegionTruth, GroundTruthSystem, SimResult, SweepCell
from .generate import (  # [V1] pure generator (imports no mechanism)
    SynChain,
    SynDomain,
    Driver,
    NullRegion,
    SyntheticSystemSpec,
    GeneratedSystem,
    generate_system,
    build_per_run_frame,
    region_path,
    frames_digest,
)

__version__ = "0.2.0+v1"

__all__ = [
    "make_rng",
    "spawn_seeds",
    "RegionTruth",
    "GroundTruthSystem",
    "SimResult",
    "SweepCell",
    # [V1]
    "SynChain",
    "SynDomain",
    "Driver",
    "NullRegion",
    "SyntheticSystemSpec",
    "GeneratedSystem",
    "generate_system",
    "build_per_run_frame",
    "region_path",
    "frames_digest",
    "__version__",
]
