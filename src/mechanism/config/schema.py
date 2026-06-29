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
