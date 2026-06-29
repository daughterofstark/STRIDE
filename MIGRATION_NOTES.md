# Migration notes — M0 (v5 → package)

**Goal:** make v5 importable and testable with **zero** change to scientific behaviour.
Every change below is structural or import-time only.

## What changed
1. **Import-time execution removed.** v5 parsed `argparse` and ran the whole
   analysis at import. The main loop (v5 lines 1909–2274) is moved verbatim into
   `_legacy.run_pipeline(config)`; `cli.py` reproduces the original argparse
   interface and calls it. Risk R7 (roadmap) is closed by the golden harness.
2. **Globals → Config.** The argparse-derived module globals
   (`TRIPLICATES_DIR, RUN_DIRS, PROTEIN_WLIST, SUBSAMPLE_STRIDE, FRAME_PS,
   ANALYSIS_FRAME_PS, DO_SENSITIVITY, FORCE_DYN_TRIAD, SKIP_MSA, GLOBAL_SEED,
   POVME_EXE`) are now set from a `Config` dataclass at the **start of**
   `run_pipeline`, with identical defaults. Functions still read them as module
   globals, so behaviour is unchanged for the default config.
3. **Pure helpers extracted verbatim.** `fdr_bh, pearson_both, block_conv_sem,
   xcorr_fn, detect_near_closure_events, canon_label` now live in
   `statistics/` and `effects/`. `_legacy` imports them (single source of truth).
   Source-identity tests assert these are byte-identical to v5.
4. **MDAnalysis import guarded.** `_legacy` wraps `import MDAnalysis` in
   try/except so the package imports without it (enabling MDAnalysis-free CI for
   the pure-stats tests). This is import-time only; at runtime the pipeline
   requires MDAnalysis exactly as before. If absent, `mda is None` and MD steps
   raise on call — same practical requirement as v5.

## What did NOT change
- Geometry, POVME invocation, distance-matrix construction, offset/triad
  detection, domain selection, all per-run and cross-protein plotting, the
  triplicate mean±SD summary, and every output filename/column. **Frozen.**
- No effective-N, bootstrap, hierarchy, variance components, reproducibility
  coefficient, or resolution gate — those are M1+ and intentionally absent.

## How parity is guaranteed
- **Static:** source-identity tests compare extracted functions against
  `reference/v5_final_code_piece.py` (must be string-identical).
- **Dynamic (local):** `capture_golden.py` snapshots ORIGINAL-v5 outputs; the
  golden tests run the refactored pipeline on a copy of the data and assert CSVs
  match within 1e-9 and figures within a pixel tolerance.
- CI runs the static + functional suite on every push (no data needed); the
  dynamic suite skips unless `MECHANISM_DATA_DIR` and fixtures are present.

## Verified in-sandbox at build time
- Package compiles and imports without MDAnalysis.
- 13/13 parity + functional tests pass, including 6 source-identity checks.

---
## M1 addendum (effective sample size)
- New module `statistics/neff.py` (IAT, N_eff, corrected SE, column helper).
- `_legacy.run_pipeline` gains one call appending `tau_int, n_eff, neff_status,
  theta_se` to each `{proj}_correlations_v5.csv`. Existing columns unchanged.
- Golden-CSV test relaxed from exact-column-set to golden⊆produced (additive
  columns), per roadmap risk R1.
- No figures changed. See REGRESSION_REPORT.md and M1_IMPLEMENTATION_NOTES.md.

---
## M2 addendum (block-bootstrap CIs)
- New module `statistics/bootstrap.py` (circular/stationary block bootstrap,
  block-length selection from M1's tau_int, Fisher-z percentile CIs, fallbacks).
- `_legacy.run_pipeline` gains one call appending six bootstrap columns after the
  M1 block. Existing and M1 columns unchanged; no figures changed.
- Default bootstrap **B=1000** (configurable). Deviation from roadmap "moving
  block" to **circular block bootstrap** default — justified in
  M2_IMPLEMENTATION_NOTES.md.
- See REGRESSION_REPORT_M2.md.

---
## M3 addendum (biological hierarchy)
- New `hierarchy/` package + `config/hierarchy_schema.py` + bundled
  `configs/denv_hierarchy.yaml`. Structural groupings only; no inference.
- `_legacy.run_pipeline` gains one (try/except-guarded) call appending
  `chain, domain, motif, secondary_structure, region_id`. `Config` gains
  `hierarchy_config`. Existing + M1/M2 columns unchanged; no figures changed.
- New dependency: PyYAML (configs may also be JSON, stdlib-only).
- See HIERARCHY.md, M3_IMPLEMENTATION_NOTES.md, REGRESSION_REPORT_M3.md.
