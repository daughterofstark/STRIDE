"""Secondary-structure assignment (milestone M3).

Pluggable by design — DSSP is never hard-coded or required. Assignment is an
interface with interchangeable backends:

* :class:`NullAssigner`     — everything ``UNKNOWN`` (default; no dependencies);
* :class:`MappingAssigner`  — from a precomputed ``canonical -> type`` mapping
  (e.g. DSSP/STRIDE output supplied via configuration);
* an external/DSSP backend can be added later behind the same interface.

Only structural labels are produced; no inference.
"""
from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from .residue import Residue


class SSEType(str, Enum):
    HELIX = "helix"
    SHEET = "sheet"
    TURN = "turn"
    LOOP = "loop"
    COIL = "coil"
    UNKNOWN = "unknown"


# One-letter codes (DSSP-style) mapped to coarse types; backend-agnostic.
_ONE_LETTER = {
    "H": SSEType.HELIX, "G": SSEType.HELIX, "I": SSEType.HELIX,
    "E": SSEType.SHEET, "B": SSEType.SHEET,
    "T": SSEType.TURN, "S": SSEType.LOOP, "P": SSEType.LOOP,
    "C": SSEType.COIL, "-": SSEType.COIL, " ": SSEType.COIL,
}


def coerce_sse(value) -> SSEType:
    """Coerce a one-letter code or type name to an :class:`SSEType`."""
    if isinstance(value, SSEType):
        return value
    v = str(value)
    if v in _ONE_LETTER:
        return _ONE_LETTER[v]
    try:
        return SSEType(v.lower())
    except ValueError:
        return SSEType.UNKNOWN


@runtime_checkable
class SSEAssigner(Protocol):
    """Interface: assign a secondary-structure type to each residue."""

    def assign(self, residues: list[Residue]) -> dict[str, SSEType]:
        ...


class NullAssigner:
    """Default backend: every residue is ``UNKNOWN`` (no external dependency)."""

    def assign(self, residues: list[Residue]) -> dict[str, SSEType]:
        return {r.key: SSEType.UNKNOWN for r in residues}


class MappingAssigner:
    """Assign from a precomputed ``canonical id -> type/one-letter`` mapping."""

    def __init__(self, assignments: dict[int, str]):
        self._a = {int(k): coerce_sse(v) for k, v in assignments.items()}

    def assign(self, residues: list[Residue]) -> dict[str, SSEType]:
        return {r.key: self._a.get(r.canonical, SSEType.UNKNOWN) for r in residues}


def make_assigner(spec) -> SSEAssigner:
    """Build an assigner from a :class:`SecondaryStructureSpec`."""
    method = getattr(spec, "method", "none")
    if method == "mapping":
        return MappingAssigner(getattr(spec, "assignments", {}) or {})
    # 'dssp' or unknown methods fall back to Null here; an external DSSP backend
    # can be registered later without changing callers.
    return NullAssigner()
