"""Residue mapping infrastructure (milestone M3).

Maps structural (file) residue identifiers to canonical ids, supporting:
insertion codes, missing residues, alternative numbering schemes (identity /
offset / explicit), multiple chains, and an alignment-map hook for future
cross-family comparison. Comparative analyses are NOT implemented here — only
the mapping infrastructure they will later use.
"""
from __future__ import annotations

import logging
from typing import Optional

from ..config.hierarchy_schema import ChainSpec, NumberingSpec
from .residue import Residue

logger = logging.getLogger(__name__)


class ResidueMapper:
    """Resolve file residue ids to canonical ids and build :class:`Residue` objects.

    Parameters
    ----------
    numbering : NumberingSpec
        Numbering scheme (identity / offset / explicit).
    chains : sequence of ChainSpec, optional
        Chain definitions by canonical range; used to assign a residue to a
        chain after canonical resolution.
    """

    def __init__(self, numbering: NumberingSpec, chains=()):
        self.numbering = numbering
        self.chains = tuple(chains)
        # per-chain alignment override: chain -> {file_resid: canonical}
        self._alignment: dict[str, dict[int, int]] = {}

    # ── numbering ────────────────────────────────────────────────────────────
    def to_canonical(self, file_resid: int, chain: Optional[str] = None) -> Optional[int]:
        """Map a file residue id to canonical, or ``None`` if unmapped (missing)."""
        if chain is not None and chain in self._alignment:
            return self._alignment[chain].get(int(file_resid))
        s = self.numbering.scheme
        if s == "identity":
            return int(file_resid)
        if s == "offset":
            return int(file_resid) - int(self.numbering.offset)
        if s == "explicit":
            return self.numbering.explicit_map.get(int(file_resid))
        logger.warning("unknown numbering scheme %r; using identity", s)
        return int(file_resid)

    # ── chain assignment ──────────────────────────────────────────────────────
    def chain_of(self, canonical: Optional[int], fallback: str = "unknown") -> str:
        """Return the chain whose canonical range contains ``canonical``."""
        if canonical is None:
            return fallback
        for c in self.chains:
            lo, hi = c.canonical_range
            if lo <= canonical <= hi:
                return c.name
        return fallback

    # ── alignment hook (infrastructure for later cross-family mapping) ────────
    def set_alignment_map(self, chain: str, file_to_canonical: dict[int, int]) -> None:
        """Install an explicit per-chain file->canonical map (e.g. from an MSA).

        Provided now so later milestones can map homologous proteins onto a
        shared canonical axis. No comparison is performed here.
        """
        self._alignment[chain] = {int(k): int(v) for k, v in file_to_canonical.items()}

    # ── construction ──────────────────────────────────────────────────────────
    def build_residue(self, file_resid: int, resname: str, *,
                      chain: Optional[str] = None, insertion: str = "") -> Residue:
        """Build a :class:`Residue`, resolving canonical id and chain."""
        canonical = self.to_canonical(file_resid, chain=chain)
        ch = chain if chain is not None else self.chain_of(canonical)
        return Residue(chain=ch, file_resid=int(file_resid), canonical=canonical,
                       resname=str(resname), insertion=str(insertion or ""))
