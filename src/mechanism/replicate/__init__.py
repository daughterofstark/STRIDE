"""Replicate-level aggregation: resolution profile, gate, mechanism assembly (M5)."""
from .aggregator import (
    GateConfig,
    Mechanism,
    build_profiles,
    gate_profile,
    run_aggregation,
)
from .orchestrate import (  # [M6]
    aggregate_from_rundirs,
    run_aggregation_tail,
    gate_config_from,
)

__all__ = [
    "GateConfig",
    "Mechanism",
    "build_profiles",
    "gate_profile",
    "run_aggregation",
    "aggregate_from_rundirs",
    "run_aggregation_tail",
    "gate_config_from",
]
