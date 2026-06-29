"""Clean entry point. M0: delegates to the lifted v5 main loop."""
from .config import Config
from . import _legacy

def run_pipeline(config: Config | None = None):
    return _legacy.run_pipeline(config or Config())
