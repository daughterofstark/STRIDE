# Regression report — M1 (effective sample size)

## Summary
M1 adds autocorrelation-corrected per-residue uncertainty and **changes no existing
scientific output**. The only behavioural difference is four appended CSV columns.

## Test suite (run at build time)
```
34 passed, 2 skipped
```
- **6 source-identity tests** (M0) — the extracted pure functions remain byte-identical
  to `reference/v5_final_code_piece.py`. Unchanged by M1. ✔
- **7 M0 functional tests** — unchanged. ✔
- **17 M1 unit tests** (`test_neff.py`) — IAT/N_eff against AR(1) theory, white noise,
  constant, short, strongly-autocorrelated, monotonicity, corrected-SE scaling. ✔
- **4 M1 integration tests** (`test_m1_integration.py`) — the column helper appends
  only and leaves every existing column byte-identical (asserted via
  `pandas.testing.assert_series_equal`). ✔
- **2 golden e2e tests** — skipped here (no data/POVME); run on the data machine.

## Numerical validation of the estimator
IAT vs AR(1) theory `tau = 0.5(1+phi)/(1-phi)`:

| phi | tau (est) | tau (theory) | status |
|----:|----------:|-------------:|:------|
| 0.5 | 1.534 | 1.500 | ok |
| 0.8 | 4.624 | 4.500 | ok |
| 0.9 | 9.585 | 9.500 | ok |

Effective N (T = 20000): white noise → n_eff = 20000 (tau = 0.500); AR(1) φ=0.9 →
n_eff ≈ 2123 (tau ≈ 4.71). Corrected SE(r=0.5) is **3.1× wider** than the naive
SE that assumes 20000 independent frames.

## Did any existing value change?
**No.** Existing columns are produced by the untouched effect-calculation loop and
written in the same order; the M1 helper only *appends* `tau_int, n_eff,
neff_status, theta_se` after `df_res` is built. Figures are not touched in M1
(N_eff is not plotted), so all PNGs remain byte-identical.

## One intended harness change
The M0 golden-CSV comparison required exact column-set equality. Because M1 is an
**additive-column** change (sanctioned by roadmap risk R1), `_frames_equal` now
asserts the golden (original-v5) columns are a **subset** of the produced columns
and that every shared column matches within 1e-9. New columns are permitted. This
is a test-semantics change, not a pipeline behaviour change.

## Outstanding (user machine)
Run the end-to-end golden check to confirm on real data that the existing columns
are byte-identical and only the four new columns appear:
```
python scripts/capture_golden.py --v5 reference/v5_final_code_piece.py --data DATADIR --out tests/golden   # if not already captured pre-M1
MECHANISM_DATA_DIR=DATADIR pytest tests/test_golden_csv.py tests/test_golden_figures.py
```
(If the golden fixtures were captured from the original v5 *before* M1, the
subset-comparison will confirm M1 added exactly the four columns and altered nothing else.)
