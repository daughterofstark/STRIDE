# M2 implementation notes — block-bootstrap confidence intervals

Implements the block-bootstrap companion to M1's analytic interval
(MATHEMATICAL_SPECIFICATION.md §2.1). Uncertainty only — the correlation effect
is never recomputed. Nothing from M3+ is present.

## Method choice (justified deviation from the roadmap)
The roadmap said "moving-block bootstrap". The default is instead the **circular
block bootstrap (CBB)**, with **stationary bootstrap (SB)** selectable:
- **CBB over plain MBB:** equilibrium MD is assumed stationary; wrapping the
  series removes MBB's endpoint under-weighting and gives every frame equal
  resampling weight (Politis & Romano 1992).
- **block method over SB by default:** at the optimal block length, CBB/MBB have
  lower variance-estimation MSE than SB (Lahiri 1999).
- **SB available:** its geometric block lengths are more robust to block-length
  misspecification, which matters because M1's `tau_int` is noisy on short MD.
This is a refinement within the block-bootstrap family, not a scope change.

## Algorithm
1. Block length `L = round(2 * tau_int)` (one+ decorrelation time), from M1's
   `tau_int`; require >= 10 blocks else fall back.
2. Resample `B` index sequences of length `n` from circular (or stationary)
   blocks; apply the **same** indices to volume and the residue distance so the
   cross-correlation is preserved (this is a cross-statistic).
3. `r_b = corr(V[idx], d[idx])` per replicate (vectorised).
4. CI on the **Fisher-z** scale: `tanh(percentile(atanh(r_b), [a/2, 1-a/2]))`
   (improves coverage for a bounded, skewed statistic); SE = `std(r_b)`.

## Assumptions
(Weak) stationarity of the analysed trajectory segment; the block length spans
the dependence range; `B` large enough for stable tail percentiles (default
1000 for 95% CIs).

## Numerical safeguards (never emit a misleading CI)
- constant / near-constant `V` or `d` -> `degenerate` (NaN bounds).
- `n < 20` or `< 10` usable blocks -> `fisher_neff` fallback (M1 interval).
- pathological resamples (>50% non-finite `r_b`) -> `fisher_neff` fallback.
- `r_b` clipped before `atanh`; every residue carries `bootstrap_method`.

## Complexity
Time `O(p/g * B * n)` after index sharing (`g` = residues per block-length
group); memory `O(B * n)` per group (~40 MB at B=1000, n=1001). Index matrices
and the volume gather are computed once per distinct block length.

## Determinism
Index matrices use child seeds spawned from the pipeline seed via
`SeedSequence([seed, L])`, so CIs are reproducible and order-independent.

## Outputs (appended columns)
`theta_bootstrap_se, theta_bootstrap_ci_lower, theta_bootstrap_ci_upper,
bootstrap_method, bootstrap_block_length, bootstrap_replicates`.

## Integration
`_legacy.run_pipeline`: one call `attach_bootstrap_ci(df_res, all_res, dm,
volumes[:n_frames], seed=GLOBAL_SEED)` immediately after the M1 call, before
`to_csv`. The effect loop and M1 block are unchanged.

## Explicitly deferred (NOT in M2)
variance components, hierarchical models, reproducibility coefficient, gate,
hierarchy, regional aggregation, calibration, coronavirus.
