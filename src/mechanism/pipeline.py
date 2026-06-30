"""Clean entry point.

M0: delegates to the lifted v5 main loop (the validated per-run engine), which is
never modified. M6: after the engine completes, runs the additive aggregation tail
that reads the per-run CSVs from disk and writes the M5 mechanism reports
(``{proj}_profile.csv`` / ``{proj}_mechanism.json``) alongside the existing
outputs. No existing output is altered.
"""
from .config import Config
from . import _legacy


def run_pipeline(config: Config | None = None):
    """Run the v5 engine, then emit the M5 reproducibility reports.

    Returns the mapping of written report paths (per protein). The engine's own
    outputs are produced exactly as before; the aggregation tail is additive and
    guarded so it can never break a validated run.
    """
    config = config or Config()
    _legacy.run_pipeline(config)          # unchanged engine: all existing outputs
    # M6: additive aggregation tail (reads per-run CSVs from disk; no engine state)
    try:
        from .replicate.orchestrate import run_aggregation_tail
        return run_aggregation_tail(config)
    except Exception as exc:              # never break the validated run
        print(f"  WARN [M6] aggregation tail skipped: {exc}")
        return {}
