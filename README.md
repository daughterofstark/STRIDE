# mechanism

Foundation package for reproducibility-gated mechanistic inference from replicate
molecular-dynamics trajectories. **This is milestone M0: a behaviour-preserving
refactor of the v5 pipeline.** No statistical methods from the mathematical
specification are implemented yet (those are M1+).

## What M0 is
- The working v5 script (`final_code_piece.py`) is now an **importable package**
  with a clean entry point — no logic changed, no outputs changed.
- Pure numerical helpers are extracted into testable modules and proven
  **byte-identical** to v5 (source-identity tests).
- A **golden regression harness** guards every future change.

## Install
```bash
pip install -e .            # core (incl. MDAnalysis) for running the pipeline
pip install -e ".[dev]"     # + pytest, pillow for the test suite
```

## Run (identical behaviour to v5)
```bash
mechanism /path/to/TRIPLICATES --stride 20 --frame-ps 10
# or
python -c "from mechanism import Config, run_pipeline; run_pipeline(Config(base_dir='/path/to/TRIPLICATES'))"
```
CLI flags mirror v5 exactly (`--proteins --stride --frame-ps --sensitivity --dynamic-triad --no-msa --seed`).

## Tests
```bash
pytest tests/test_stats_parity.py            # runs anywhere (no MDAnalysis needed)
# End-to-end golden parity (needs data + POVME + MDAnalysis):
python scripts/capture_golden.py --v5 reference/v5_final_code_piece.py --data DATADIR --out tests/golden
MECHANISM_DATA_DIR=DATADIR pytest tests/test_golden_csv.py tests/test_golden_figures.py
```

## Layout
```
src/mechanism/
  __init__.py        Config, run_pipeline
  pipeline.py        clean entry point -> _legacy.run_pipeline
  cli.py             v5-parity command line
  config/schema.py   Config dataclass (v5 defaults)
  statistics/        fdr, correlation, convergence, xcorr, events  (pure, extracted verbatim)
  effects/labels.py  canon_label
  _legacy.py         the lifted v5 engine (MD/POVME/plots) + run_pipeline
reference/           the original v5 source (for source-identity tests)
scripts/             capture_golden.py
tests/               parity + golden harness
```
See `MIGRATION_NOTES.md` for exactly what changed and why.
