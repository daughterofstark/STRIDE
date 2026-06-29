"""Biological hierarchy data model (M3): structural groupings only, no inference."""
from .residue import Residue
from .region import Region
from .mapping import ResidueMapper
from .domain import Group, GroupResolver
from .secondary_structure import (
    SSEType, SSEAssigner, NullAssigner, MappingAssigner, make_assigner, coerce_sse,
)
from .hierarchy import (
    BiologicalHierarchy, build_hierarchy, attach_structural_metadata,
)

__all__ = [
    "Residue", "Region", "ResidueMapper", "Group", "GroupResolver",
    "SSEType", "SSEAssigner", "NullAssigner", "MappingAssigner",
    "make_assigner", "coerce_sse",
    "BiologicalHierarchy", "build_hierarchy", "attach_structural_metadata",
]
