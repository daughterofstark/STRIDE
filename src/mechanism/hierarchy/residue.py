"""Residue data model for the biological hierarchy (milestone M3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Residue:
    """A single residue with structural identity and canonical numbering.

    Attributes
    ----------
    chain : str
        Chain name (from configuration; family-agnostic).
    file_resid : int
        Residue id as it appears in the structure/topology.
    canonical : int or None
        Canonical id after numbering resolution; ``None`` if unmapped (missing).
    resname : str
        Residue (amino-acid) name.
    insertion : str
        Insertion code (PDB), empty if none.
    """

    chain: str
    file_resid: int
    canonical: Optional[int]
    resname: str
    insertion: str = ""

    @property
    def key(self) -> str:
        """Stable, unique key across chains and insertion codes."""
        return f"{self.chain}:{self.file_resid}{self.insertion}"
