"""Configuration schema for the biological hierarchy (milestone M3).

Entirely protein-family agnostic: no virus, chain, domain, or catalytic residue
is hard-coded. A new protein family is supported by supplying a configuration
file (YAML or JSON); the source code never changes. DENV1-4 is the first
implementation; coronavirus is the first validation (later milestone).

This module defines *structure only*. It performs no inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _expand(spec) -> list[int]:
    """Expand a residue spec into a list of canonical ids.

    Accepts a list of ints, ``[lo, hi]`` inclusive ranges written as
    ``{"range": [lo, hi]}``, or a mixed list of ints and such range dicts.
    """
    out: list[int] = []
    if spec is None:
        return out
    if isinstance(spec, dict) and "range" in spec:
        lo, hi = spec["range"]
        return list(range(int(lo), int(hi) + 1))
    for item in spec:
        if isinstance(item, dict) and "range" in item:
            lo, hi = item["range"]
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return out


@dataclass(frozen=True)
class NumberingSpec:
    """How structural (file) residue ids map to canonical ids.

    scheme:
        ``identity`` -> canonical == file id;
        ``offset``   -> canonical == file id - offset;
        ``explicit`` -> canonical == explicit_map[file id] (else missing).
    """

    scheme: str = "identity"
    offset: int = 0
    explicit_map: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainSpec:
    """A chain, identified by an inclusive canonical residue range.

    Family-agnostic: e.g. a cofactor on negative canonical numbers and a
    protease on positive numbers are two chains; a single-chain protein has one.
    """

    name: str
    canonical_range: tuple[int, int]
    structural_id: Optional[str] = None  # optional PDB/segid hint


@dataclass(frozen=True)
class GroupSpec:
    """A named residue group (domain or motif): canonical ids, optional chain."""

    name: str
    residues: tuple[int, ...]
    chain: Optional[str] = None
    order: int = 0  # config order, used for deterministic overlap tie-breaks


@dataclass(frozen=True)
class CatalyticResidue:
    canonical: int
    resname: Optional[str] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class SecondaryStructureSpec:
    """How secondary structure is assigned. Never hard-codes DSSP.

    method:
        ``none``    -> all residues UNKNOWN (default; no dependency);
        ``mapping`` -> use ``assignments`` (canonical id -> one-letter/type);
        ``dssp``    -> compute via an external assigner if available (optional).
    """

    method: str = "none"
    assignments: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchyConfig:
    """Complete biological configuration for one protein family."""

    name: str
    levels: tuple[str, ...] = (
        "complex", "protein", "chain", "domain", "motif",
        "secondary_structure", "residue",
    )
    complex_name: str = "complex"
    protein_name: str = "protein"
    numbering: NumberingSpec = field(default_factory=NumberingSpec)
    chains: tuple[ChainSpec, ...] = ()
    domains: tuple[GroupSpec, ...] = ()
    motifs: tuple[GroupSpec, ...] = ()
    catalytic_residues: tuple[CatalyticResidue, ...] = ()
    secondary_structure: SecondaryStructureSpec = field(
        default_factory=SecondaryStructureSpec)
    aliases: dict[str, str] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────
    @staticmethod
    def from_dict(d: dict) -> "HierarchyConfig":
        num = d.get("numbering", {}) or {}
        numbering = NumberingSpec(
            scheme=num.get("scheme", "identity"),
            offset=int(num.get("offset", 0)),
            explicit_map={int(k): int(v) for k, v in (num.get("explicit_map") or {}).items()},
        )
        chains = tuple(
            ChainSpec(name=c["name"],
                      canonical_range=(int(c["canonical_range"][0]), int(c["canonical_range"][1])),
                      structural_id=c.get("structural_id"))
            for c in d.get("chains", []) or []
        )

        def _groups(key):
            return tuple(
                GroupSpec(name=g["name"], residues=tuple(_expand(g.get("residues"))),
                          chain=g.get("chain"), order=i)
                for i, g in enumerate(d.get(key, []) or [])
            )

        ss = d.get("secondary_structure", {}) or {}
        sspec = SecondaryStructureSpec(
            method=ss.get("method", "none"),
            assignments={int(k): str(v) for k, v in (ss.get("assignments") or {}).items()},
        )
        cats = tuple(
            CatalyticResidue(canonical=int(c["canonical"]),
                             resname=c.get("resname"), role=c.get("role"))
            for c in d.get("catalytic_residues", []) or []
        )
        return HierarchyConfig(
            name=d.get("name", "protein"),
            levels=tuple(d.get("levels", HierarchyConfig.levels)),
            complex_name=d.get("complex_name", "complex"),
            protein_name=d.get("protein_name", "protein"),
            numbering=numbering, chains=chains,
            domains=_groups("domains"), motifs=_groups("motifs"),
            catalytic_residues=cats, secondary_structure=sspec,
            aliases={str(k): str(v) for k, v in (d.get("aliases") or {}).items()},
        )

    @staticmethod
    def load(path: str | Path) -> "HierarchyConfig":
        """Load from a ``.yaml``/``.yml`` (lazy PyYAML) or ``.json`` file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # lazy; only needed for YAML configs
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return HierarchyConfig.from_dict(data)
