# M1 verification report

**Verdict: M1 satisfies the roadmap.** No existing v5 numerical output or figure
code changed; the only difference is four appended uncertainty columns. The one
check that cannot run in this environment (end-to-end byte-identity on real
trajectories) is specified below for the data machine.

## 1. Regression — existing outputs unchanged (proven at source level)

**Function-level audit** (`_legacy` vs `reference/v5_final_code_piece.py`):
- v5 functions: 38 → 6 extracted to modules in M0 (each separately proven
  byte-identical by the source-identity tests) + **32 byte-identical in `_legacy`**.
- **0 functions differ. 0 missing.** Only new function: `run_pipeline`.
- This includes every `plot_*` and `fig_*` function ⇒ **figure-generating code is
  unchanged**, so figures are identical for identical inputs.

**Main-loop audit** (`run_pipeline` body vs v5 lines 1909–2274):
- **+10 lines, −0 lines, 0 changed.** The 10 added lines are exactly the M1
  comment block plus the single `attach_effective_sample_size(...)` call,
  inserted *after* `df_res` is built and *before* `to_csv`. The effect loop, the
  POVME calls, the plotting calls and the triplicate tail are untouched.

**Additive-column demonstration** (representative table, 9 v5 columns):
- Existing columns after the M1 helper: **all byte-identical** (verified
  column-by-column).
- Added, in order: `tau_int, n_eff, neff_status, theta_se`. Nothing else.

**Numerical differences discovered in existing outputs: NONE.**

## 2. Unit test summary
`34 passed, 2 skipped` (skips = end-to-end golden tests; no data/POVME here).
- 6 source-identity (M0 pure functions) — still pass under M1.
- 7 M0 functional.
- 15 M1 `test_neff.py` — AR(1) theory, white noise, constant, short,
  strong-autocorrelation, monotonicity, corrected-SE scaling.
- 4 M1 `test_m1_integration.py` — additive-only, column order, value validity,
  alignment under the canon-resid sort.

Estimator accuracy (AR(1), `tau=0.5(1+phi)/(1-phi)`): 0.5→1.534/1.500,
0.8→4.624/4.500, 0.9→9.585/9.500 (≤3% error). White noise → N_eff = T;
AR(1) φ=0.9 → N_eff 20000→~2123, corrected SE **3.1×** the naive SE.

## 3. Code coverage
| Module | Cover |
|---|---|
| statistics/neff.py | **94%** |
| statistics/{correlation,xcorr} | 100% |
| statistics/{fdr,convergence} | 95% / 94% |
| statistics/events | 71% |
| config, effects, __init__ | 100% |
| _legacy.py | **7%** |

The M1 code is 94% covered. `_legacy.py` is low because its 1,191-statement MD
pipeline cannot execute without MDAnalysis + POVME + trajectories; it is exercised
by the end-to-end golden suite on the data machine, not in CI. The 6 uncovered
`neff.py` lines are defensive branches: the empty-input constant guard and the two
`undersampled_capped` cap paths (reached only by near-unit-root short series).
These are correct by inspection; targeted tests can be added on request.

## 4. Runtime benchmark
M1 added cost (1001 frames × 250 residues = one protein-run): **~58 ms**
(~233 µs/residue). Full study (4 serotypes × 3 runs): **~0.70 s** total —
negligible beside POVME/MD. IAT: ~154 µs/call. Scaling vs frames:
1001→57 ms, 5000→259 ms, 20000→1016 ms (≈linear).

## 5. The one check that must run on the data machine
Source-level proof above establishes that existing computations and figure code
are unchanged. The final byte-identity confirmation on real trajectory data is:
```
# golden fixtures should be captured from ORIGINAL v5 (pre-M1):
python scripts/capture_golden.py --v5 reference/v5_final_code_piece.py --data DATADIR --out tests/golden
MECHANISM_DATA_DIR=DATADIR pytest tests/test_golden_csv.py tests/test_golden_figures.py
```
Expected: CSV existing columns match within 1e-9 with exactly four new columns
added; all PNGs within pixel tolerance. This is the only outstanding gate before M2.
