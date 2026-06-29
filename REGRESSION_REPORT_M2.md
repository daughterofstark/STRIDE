# Regression report — M2 (block-bootstrap CIs)

## Verdict
M2 adds six bootstrap-uncertainty columns and **changes no existing output**.

## Existing outputs unchanged (source-level proof)
- Function audit `_legacy` vs `reference/v5_final_code_piece.py`: **0 v5
  functions differ** (excluding the 6 M0-extracted ones, separately proven
  byte-identical); only new function is `run_pipeline`.
- M2 footprint in `_legacy`: **one import + one call** (lines 35, 1839–1844);
  inserted after the M1 block, before `to_csv`. Effect loop, POVME, plotting,
  triplicate tail, and the M1 block are untouched.
- M1 columns (`tau_int, n_eff, neff_status, theta_se`) are read-only inputs to
  M2 and are not modified (asserted by `test_attach_appends_only_and_preserves_existing`).
- Figures: no plotting code touched -> identical.

## Tests
`51 passed, 2 skipped` (skips = end-to-end golden, no data here).
- M0: 13 (incl. 6 source-identity). M1: 21. **M2: 17.**
- M2 coverage validation: empirical 95%-CI coverage within [0.86, 0.995] for
  both independent (phi=0) and autocorrelated (phi=0.7) data.
- Bootstrap vs Fisher: ~independent data -> CI width within 0.6–1.6x Fisher;
  autocorrelated data -> bootstrap CI strictly **wider** than naive-N Fisher
  (the intended correction).
- Fallbacks: constant -> `degenerate` (NaN, no fake CI); very short ->
  `fisher_neff`. Determinism verified (same seed -> identical CIs).

## Coverage (code)
`bootstrap.py` 96%; all statistics modules 94–100%; `_legacy.py` 7% (its MD
pipeline needs MDAnalysis+POVME+data; covered by the golden suite on the data
machine).

## Runtime (after optimisation)
Default **B=1000**: ~4.7 s per protein-run (1001 frames x 250 residues),
~**70 s** for the full 12-run study; ~41 MB peak. (B=2000 ~14 s/run, ~168 s.)
Optimisation applied: residues sharing a block length share one index matrix and
one volume gather (computed once per distinct L), and the default was lowered
from 2000 to 1000 (adequate for 95% percentile CIs). Frame scaling is roughly
linear at MD scale; pathological random-walk test inputs inflate L and cost.

## Numerical differences in existing values
**None.** Only additive columns. (M1's one test-semantics change — golden CSV
subset comparison — already covers additive columns; M2 needs no further harness
change.)

## Outstanding (data machine)
End-to-end byte-identity of existing columns + figures on real trajectories via
the golden harness (`MECHANISM_DATA_DIR=... pytest tests/test_golden_csv.py
tests/test_golden_figures.py`).
