"""The single validation->production bridge (milestone V1).

This is the **only** non-test module in ``validation/`` permitted to import
``mechanism``, and it imports **only** the public hierarchy schema
(``mechanism.config.hierarchy_schema``). It translates a pure
:class:`validation.generate.SyntheticSystemSpec` into a production
:class:`mechanism.config.hierarchy_schema.HierarchyConfig` so that generated
per-run frames round-trip through the frozen production estimator
(``aggregate_reproducibility`` / ``run_aggregation``).

Separation contract (V1, tightened from V0)
-------------------------------------------
* ``generate.py`` stays ``mechanism``-free; only this module bridges.
* This module touches **only** ``config.hierarchy_schema`` — never
  ``build_hierarchy``, never ``_legacy``, never any underscore-prefixed name.
  The estimator rebuilds the hierarchy internally from the frames; validation
  never calls ``build_hierarchy`` itself.
* ``mechanism`` never imports ``validation`` (unchanged from V0).

The static separation test (``validation/tests/test_separation.py``) enforces
that the set of validation modules importing ``mechanism`` is a subset of
``{adapters.py}`` and that only public names are used.
"""
from __future__ import annotations

# Public API only. No underscore-prefixed production internals.
from mechanism.config.hierarchy_schema import (
    HierarchyConfig, ChainSpec, GroupSpec, NumberingSpec,
)

from .generate import SyntheticSystemSpec


def to_hierarchy_config(spec: SyntheticSystemSpec) -> HierarchyConfig:
    """Build a production :class:`HierarchyConfig` from a synthetic spec.

    Maps ``SynChain -> ChainSpec`` and ``SynDomain -> GroupSpec`` and sets the
    numbering to ``offset`` with the spec's offset (so canonical ids recover as
    ``file_resid - offset``, exactly as the estimator expects). Levels are carried
    through verbatim (coarse->fine), so the reconstructed nested partitions match
    the ground-truth scale indices.

    The generated frames already carry canonical ids and chain labels; this config
    only needs to reproduce the chain ranges, domain memberships, and level order
    the estimator uses to rebuild regions.
    """
    chains = tuple(
        ChainSpec(name=c.name,
                  canonical_range=(int(c.canonical_range[0]),
                                   int(c.canonical_range[1])))
        for c in spec.chains
    )
    domains = tuple(
        GroupSpec(name=d.name, residues=tuple(int(r) for r in d.residues),
                  chain=d.chain, order=i)
        for i, d in enumerate(spec.domains)
    )
    return HierarchyConfig(
        name=spec.name,
        levels=tuple(spec.levels),
        numbering=NumberingSpec(scheme="offset", offset=int(spec.offset)),
        chains=chains,
        domains=domains,
        motifs=(),
    )
