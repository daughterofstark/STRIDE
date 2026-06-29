"""Resolve which hierarchy configuration to use (data selection, not biology).

No biological constants live here. An explicit path always wins. Otherwise a
bundled family configuration is selected by protein name if one exists; failing
that, a generic single-chain configuration is returned so that *any* protein is
supported without source changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config.hierarchy_schema import HierarchyConfig, ChainSpec, NumberingSpec

logger = logging.getLogger(__name__)

_BUNDLED = Path(__file__).resolve().parent.parent / "configs"


def generic_config(name: str) -> HierarchyConfig:
    """A minimal, family-agnostic config: one chain spanning all residues."""
    return HierarchyConfig(
        name=name,
        chains=(ChainSpec(name="A", canonical_range=(-10 ** 9, 10 ** 9)),),
        numbering=NumberingSpec(scheme="offset", offset=0),
    )


def resolve_hierarchy_config(proj: str, explicit_path: str | None = None) -> HierarchyConfig:
    """Return a :class:`HierarchyConfig` for ``proj``.

    Order: explicit path -> ``configs/{proj}_hierarchy.yaml`` ->
    bundled DENV config for ``DENV*`` -> generic fallback.
    """
    if explicit_path:
        return HierarchyConfig.load(explicit_path)
    named = _BUNDLED / f"{proj}_hierarchy.yaml"
    if named.exists():
        return HierarchyConfig.load(named)
    if proj.upper().startswith("DENV"):
        denv = _BUNDLED / "denv_hierarchy.yaml"
        if denv.exists():
            cfg = HierarchyConfig.load(denv)
            # carry the specific serotype name through
            return HierarchyConfig.from_dict({**_as_dict(cfg), "name": proj})
    logger.info("no hierarchy config for %s; using generic single-chain config", proj)
    return generic_config(proj)


def _as_dict(cfg: HierarchyConfig) -> dict:
    """Round-trip a config back to a dict (used to rename without reparsing YAML)."""
    return {
        "name": cfg.name, "levels": list(cfg.levels),
        "complex_name": cfg.complex_name, "protein_name": cfg.protein_name,
        "numbering": {"scheme": cfg.numbering.scheme, "offset": cfg.numbering.offset,
                      "explicit_map": cfg.numbering.explicit_map},
        "chains": [{"name": c.name, "canonical_range": list(c.canonical_range),
                    "structural_id": c.structural_id} for c in cfg.chains],
        "domains": [{"name": g.name, "residues": list(g.residues), "chain": g.chain}
                    for g in cfg.domains],
        "motifs": [{"name": g.name, "residues": list(g.residues), "chain": g.chain}
                   for g in cfg.motifs],
        "catalytic_residues": [{"canonical": c.canonical, "resname": c.resname,
                                "role": c.role} for c in cfg.catalytic_residues],
        "secondary_structure": {"method": cfg.secondary_structure.method,
                                "assignments": cfg.secondary_structure.assignments},
        "aliases": cfg.aliases,
    }
