"""Biological hierarchy assembly and aggregation API (milestone M3).

Builds a nested, protein-family-agnostic hierarchy over residues:

    complex > protein > chain > domain > motif > secondary_structure > residue

Region identifiers are *path tuples* (coarsest..level), so the levels are nested
by construction and every level is a partition of the residues (each residue lies
in exactly one region per level). Arbitrary depth is supported: any configured
level without a known component getter receives a deterministic placeholder, so
custom levels still yield a valid nested partition.

This module produces **structural groupings only**. It performs no inference,
no aggregation of statistics, no reproducibility computation, and no gate.
``aggregate(level=...)`` returns groupings of residue keys — nothing more.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..config.hierarchy_schema import HierarchyConfig
from .domain import GroupResolver
from .mapping import ResidueMapper
from .region import Region
from .residue import Residue
from .secondary_structure import SSEAssigner, make_assigner

logger = logging.getLogger(__name__)

# Levels for which we know how to derive a path component. Unknown configured
# levels are still honoured (placeholder component) to allow arbitrary depth.
_KNOWN_ORDER = [
    "complex", "protein", "chain", "domain", "motif",
    "secondary_structure", "residue",
]

_UNASSIGNED = "unassigned"
_NO_MOTIF = "none"


class BiologicalHierarchy:
    """A nested structural hierarchy over a set of residues.

    Parameters
    ----------
    residues : sequence of Residue
        Residues with chain and canonical id already resolved (via
        :class:`ResidueMapper`).
    config : HierarchyConfig
        Family-agnostic biological configuration.
    assigner : SSEAssigner, optional
        Secondary-structure backend; defaults to the one named in ``config``.
    """

    def __init__(self, residues: Iterable[Residue], config: HierarchyConfig,
                 assigner: Optional[SSEAssigner] = None):
        self.config = config
        self.residues = list(residues)
        self._by_key = {r.key: r for r in self.residues}
        self._domain = GroupResolver(config.domains)
        self._motif = GroupResolver(config.motifs)
        assigner = assigner or make_assigner(config.secondary_structure)
        self._sse = assigner.assign(self.residues)

        # configured level order, restricted to a sensible coarse->fine ordering
        cfg_levels = [l for l in config.levels]
        self.level_order = [l for l in _KNOWN_ORDER if l in cfg_levels]
        # append any custom (unknown) levels at the end, preserving config order
        for l in cfg_levels:
            if l not in self.level_order:
                self.level_order.append(l)

        # full path component per residue, in level order
        self._path: dict[str, tuple] = {
            r.key: self._build_path(r) for r in self.residues
        }

    # ── component derivation ─────────────────────────────────────────────────
    def _component(self, level: str, r: Residue) -> str:
        if level == "complex":
            return self.config.complex_name
        if level == "protein":
            return self.config.protein_name
        if level == "chain":
            return r.chain
        if level == "domain":
            return self._domain.resolve(r.canonical, r.chain) or _UNASSIGNED
        if level == "motif":
            return self._motif.resolve(r.canonical, r.chain) or _NO_MOTIF
        if level == "secondary_structure":
            return self._sse.get(r.key).value if self._sse.get(r.key) else "unknown"
        if level == "residue":
            return r.key
        return f"({level})"  # arbitrary custom level -> placeholder

    def _build_path(self, r: Residue) -> tuple:
        return tuple(self._component(l, r) for l in self.level_order)

    # ── public API (structural groupings only) ───────────────────────────────
    @property
    def levels(self) -> list[str]:
        return list(self.level_order)

    def _level_index(self, level: str) -> int:
        try:
            return self.level_order.index(level)
        except ValueError as exc:
            raise KeyError(f"unknown level {level!r}; available: {self.level_order}") from exc

    def region_of(self, residue_key: str, level: str) -> tuple:
        """Region id (path prefix) of a residue at the requested level."""
        i = self._level_index(level)
        return self._path[residue_key][: i + 1]

    def aggregate(self, level: str = "domain") -> dict[tuple, list[str]]:
        """Group residue keys by their region at ``level`` (a partition).

        Returns groupings only — no statistics. Nested by construction: the
        keys at a coarser level are prefixes of those at a finer level.
        """
        i = self._level_index(level)
        groups: dict[tuple, list[str]] = {}
        for key, path in self._path.items():
            groups.setdefault(path[: i + 1], []).append(key)
        return groups

    def regions(self, level: str = "domain") -> list[Region]:
        """Return :class:`Region` objects for ``level``."""
        out = []
        for region_id, members in self.aggregate(level).items():
            parent = region_id[:-1] if len(region_id) > 1 else None
            out.append(Region(level=level, region_id=region_id,
                              label=region_id[-1], members=tuple(members),
                              parent_id=parent))
        return out

    def annotate(self, residue_key: str, region_level: str = "domain") -> dict:
        """Structural metadata for one residue (no statistics)."""
        path = self._path[residue_key]
        comp = dict(zip(self.level_order, path))
        ri = self._level_index(region_level)
        return {
            "chain": comp.get("chain", "unknown"),
            "domain": comp.get("domain", _UNASSIGNED),
            "motif": comp.get("motif", _NO_MOTIF),
            "secondary_structure": comp.get("secondary_structure", "unknown"),
            "region_id": "/".join(str(x) for x in path[: ri + 1]),
        }


# ── construction + pipeline helper ───────────────────────────────────────────
def build_hierarchy(md_residues, config: HierarchyConfig, *, offset: int = 0,
                    assigner: Optional[SSEAssigner] = None) -> BiologicalHierarchy:
    """Build a hierarchy from structure residues using a runtime numbering offset.

    ``md_residues`` only needs ``.resid`` and ``.resname`` attributes (e.g.
    MDAnalysis residues). Canonical id = ``resid - offset`` (the pipeline's
    universal convention); chains/domains/motifs are resolved in canonical space
    from ``config`` — so no family-specific logic lives in code.
    """
    from ..config.hierarchy_schema import NumberingSpec
    mapper = ResidueMapper(NumberingSpec(scheme="offset", offset=offset),
                           chains=config.chains)
    residues = []
    for r in md_residues:
        insertion = str(getattr(r, "icode", "") or "")
        residues.append(mapper.build_residue(int(r.resid), str(r.resname),
                                             insertion=insertion))
    return BiologicalHierarchy(residues, config, assigner=assigner)


def attach_structural_metadata(df_res, md_residues, config: HierarchyConfig, *,
                               offset: int = 0, resid_col: str = "file_resid",
                               region_level: str = "domain"):
    """Append structural metadata columns to a per-residue table (M3).

    Appends ``chain, domain, motif, secondary_structure, region_id``. Reads only;
    never modifies any existing column or any effect/uncertainty value.
    """
    hierarchy = build_hierarchy(md_residues, config, offset=offset)
    # map file_resid -> residue key (single occurrence per file id in this use)
    key_by_fid = {r.file_resid: r.key for r in hierarchy.residues}
    ann = {fid: hierarchy.annotate(key, region_level=region_level)
           for fid, key in key_by_fid.items()}

    def _col(name):
        return df_res[resid_col].map(
            lambda f: ann.get(int(f), {}).get(name, "unknown"))

    df_res["chain"] = _col("chain")
    df_res["domain"] = _col("domain")
    df_res["motif"] = _col("motif")
    df_res["secondary_structure"] = _col("secondary_structure")
    df_res["region_id"] = _col("region_id")
    return df_res
