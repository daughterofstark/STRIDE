"""Domain and motif grouping (milestone M3).

Resolves which named group (domain or motif) a residue belongs to, with
deterministic rules for overlapping and non-covering definitions. Family-
agnostic: groups are supplied by configuration as canonical residue ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config.hierarchy_schema import GroupSpec


@dataclass(frozen=True)
class Group:
    """A resolved domain/motif: name, optional chain, canonical residue set."""

    name: str
    chain: Optional[str]
    residues: frozenset
    order: int

    @staticmethod
    def from_spec(spec: GroupSpec) -> "Group":
        return Group(name=spec.name, chain=spec.chain,
                     residues=frozenset(spec.residues), order=spec.order)


class GroupResolver:
    """Map a residue to its group, resolving overlaps deterministically.

    Overlap rule: the **most specific** (smallest) matching group wins; ties are
    broken by configuration order. Coverage rule: residues in no group return
    ``None`` (the caller substitutes an ``unassigned`` placeholder so the
    hierarchy stays a complete partition).
    """

    def __init__(self, specs):
        self._groups = [Group.from_spec(s) for s in specs]

    def resolve(self, canonical: Optional[int], chain: Optional[str] = None) -> Optional[str]:
        if canonical is None:
            return None
        candidates = [
            g for g in self._groups
            if (g.chain is None or g.chain == chain) and canonical in g.residues
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda g: (len(g.residues), g.order))
        return best.name

    def names(self) -> list[str]:
        return [g.name for g in self._groups]
