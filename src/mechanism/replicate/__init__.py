"""Replicate-level aggregation: resolution profile, gate, mechanism assembly (M5)."""
from .aggregator import (
    GateConfig,
    Mechanism,
    build_profiles,
    gate_profile,
    run_aggregation,
)

__all__ = [
    "GateConfig",
    "Mechanism",
    "build_profiles",
    "gate_profile",
    "run_aggregation",
]
