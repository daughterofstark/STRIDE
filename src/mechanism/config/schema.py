"""Configuration for the pipeline. M0: mirrors v5 CLI defaults exactly."""
from dataclasses import dataclass, field
from typing import Optional, List
import os

@dataclass
class Config:
    base_dir: str = os.path.expanduser("~/Desktop/actual_final_stuff_medha")
    run_dirs: List[str] = field(default_factory=lambda: ["1st_run","2nd_run","3rd_run"])
    proteins: Optional[List[str]] = None   # whitelist; None = discover all
    stride: int = 20                       # SUBSAMPLE_STRIDE
    frame_ps: float = 10.0                 # ps per saved frame
    sensitivity: bool = False              # POVME 12/14/16 radii
    dynamic_triad: bool = False            # force per-frame triad
    no_msa: bool = False                   # skip MSA equivalence
    seed: int = 42                         # global RNG seed
    hierarchy_config: Optional[str] = None # path to a hierarchy YAML/JSON (M3)
    # M6 — gate settings consumed by the M5 aggregation tail. PROVISIONAL and
    # UNCALIBRATED: rho_star is a configured constant, not the Part IV calibrated
    # threshold; emitted mechanisms are marked calibrated=False until the
    # validation phase. Defaults mirror replicate.GateConfig.
    rho_star: float = 0.5
    alpha: float = 0.05
    coherence_threshold: float = 0.6
