"""mechanism — M0 behaviour-preserving package wrapping the v5 pipeline."""
__version__ = "0.6.0+m6"
from .config import Config
from .pipeline import run_pipeline
__all__ = ["Config", "run_pipeline"]
