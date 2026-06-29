"""Region data model (milestone M3).

A :class:`Region` is a structural grouping of residues at one hierarchy level.
Region identifiers are *paths* (tuples from coarsest to this level), which makes
the levels nested by construction: the region of a residue at a coarser level is
always a prefix of its region at a finer level. This is purely structural — no
statistics are attached.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """A structural grouping of residues at one level.

    Attributes
    ----------
    level : str
        Hierarchy level name (e.g. ``domain``).
    region_id : tuple
        Path tuple identifying the region (coarsest..this level). Unique.
    label : str
        Human-readable label (the leaf component of the path).
    members : tuple of str
        Residue keys belonging to this region.
    parent_id : tuple or None
        ``region_id`` of the parent region (one level coarser), or ``None`` at
        the root.
    """

    level: str
    region_id: tuple
    label: str
    members: tuple
    parent_id: tuple | None

    @property
    def size(self) -> int:
        return len(self.members)
