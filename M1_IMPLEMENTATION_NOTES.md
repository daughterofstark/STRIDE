# M1 implementation notes — spec → code

Implements **only** §2.1 of MATHEMATICAL_SPECIFICATION.md (within-replicate
uncertainty). Nothing from §2.2+ (variance components, reproducibility
coefficient, gate) is present.

## Mapping

| Spec (§2.1) | Code |
|---|---|
| `tau_int = 1/2 + Σ_{Δ≥1} ρ(Δ)` | `neff.integrated_autocorr_time` (Sokal windowing, FFT ACF) |
| FFT autocorrelation | `neff.autocorr_fft` (zero-padded, non-circular, normalised) |
| `N_eff = T / (2·tau_int)` | `neff.effective_sample_size`, on the product series `z(t)=(V-Ē V)(d-Ē d)` |
| Fisher SE `√((1-θ²)²/N_eff)` | `neff.corrected_standard_error` (reads existing `r`, never recomputes it) |
| append per-residue uncertainty | `neff.attach_effective_sample_size` → columns `tau_int, n_eff, neff_status, theta_se` |

## Design choices
- **Product-series IAT.** The covariance integrand `z(t)` is the series whose
  autocorrelation governs a correlation estimate's sampling variance, so
  `tau_int` is computed on `z`. (AR(1) theory is unit-tested on the estimator
  itself, where the analytic value is exact.)
- **Sokal automatic windowing** (`M ≥ 5·tau_int(M)`) — standard, stable, no tuning.
- **Safeguards.** `tau_int` floored at 0.5 (white-noise limit) and capped at
  `T/4` so `n_eff ≥ 2`; constant signals → `constant_signal`; `T < 8` →
  `short_trajectory`; non-convergent windowing → `undersampled_capped`. Every
  residue carries a `neff_status` describing which path was taken.
- **Effect field untouched.** `attach_effective_sample_size` reads `volumes`,
  the Cα distance matrix and the existing `r` column; it appends columns only.

## Integration point
`_legacy.run_pipeline`: a single call to `attach_effective_sample_size(df_res,
all_res, dm, volumes[:n_frames])` inserted *after* `df_res` is built and *before*
`to_csv`. The effect-calculation loop above it is unchanged.

## Explicitly deferred (NOT in M1)
block bootstrap & confidence intervals (M2), hierarchy (M3), variance
components / reproducibility coefficient (M4), resolution gate (M5),
calibration, coronavirus.
