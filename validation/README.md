# `validation/` — STRIDE validation & benchmarking framework (Phase 2)

This package is **completely separate** from the production `mechanism` package and
implements the validation study described in `../VALIDATION_ROADMAP.md` (milestones
V0–V8). It is a research/dev package, **not** part of the shipped `mechanism`
distribution.

## Separation invariants (must hold at every milestone)

1. `validation/` lives outside the packaged `src/` tree, so building or installing
   `mechanism` is identical with or without it (`pyproject.toml` packages only
   `src/`).
2. `mechanism` never imports `validation` — there is **no** reverse dependency.
   Deleting this directory leaves Phase 1 (M0–M6) fully functional and golden-green.
3. `validation` imports `mechanism` only through its **documented public API**
   (`mechanism.Config`, `mechanism.run_pipeline`, and the documented submodule
   functions). At V0 it imports nothing from `mechanism`.
4. Validation tests live under `validation/tests/`, outside the production
   `testpaths = ["tests"]`, so a production test run never collects them.
5. All randomness flows through `validation._seed` (seeded, reproducible, recorded).

## V0 contents

| Module | Purpose |
|---|---|
| `_seed.py` | deterministic seeded RNG layer (`make_rng`, `spawn_seeds`) |
| `types.py` | ground-truth data contract (`GroundTruthSystem`, `RegionTruth`, `SimResult`, `SweepCell`) |
| `cli.py` | skeleton CLI with placeholder subcommands |
| `tests/` | separation guard, determinism, and data-contract tests |

## V1 contents (Tier-A field-level generator)

| Module | Purpose |
|---|---|
| `generate.py` | **pure** Tier-A generator (imports no `mechanism`): recipe dataclasses (`SyntheticSystemSpec`, `SynChain`, `SynDomain`, `Driver`, `NullRegion`), `generate_system`, the production-schema frame builder `build_per_run_frame` (factored for Tier-B/V2 reuse), the `region_path` helper, and `frames_digest` |
| `adapters.py` | the **single** validation→production bridge: `to_hierarchy_config(spec)`; the only non-test module allowed to import `mechanism`, and only from the public `mechanism.config.hierarchy_schema` |
| `tests/test_generate.py`, `tests/test_adapters.py` | generator behavior, round-trip closure through the frozen production estimator, I1, determinism, hierarchy/offset correctness, pathological regimes, adapter correctness |

### What V1 plants (and what it does not)

V1 realizes the **field level** of the §2.2 random-effects model:
`theta_i^(k) = (beta + gamma^(k)) * carrier_i^(k) * sign_i + eps_i^(k)`, with
`gamma ~ N(0, tau^2)`, `eps ~ N(0, sigma^2)`, and a carrier that permutes within a
driver region (Property I1). It plants `sigma^2` **directly** (Tier A); the
autocorrelated-series derivation `sigma^2 = (1 - theta^2)^2 / N_eff` is Tier B (V2).

### Observed property of the current estimator (characterized, not hidden)

The production estimator fits the model on the folded energy
`A_en = sqrt(sum theta^2)`. Two consequences are documented and **tested as
characterizations**, not asserted as spec or theorems:

* a driver region's recovered `beta_hat` estimates `E|beta + gamma|` (exact only in
  the low-`tau`/high-SNR corner);
* a pure-null region of `m >= 2` residues with within-noise reads a positive
  `rho_hat` that **grows with region size** (`test_null_region_rho_grows_with_size…`).

V1 therefore asserts the *achievable* property — a driver region **separates** from
equal-size null regions — rather than the idealized "nulls read as zero". Calibrating
`rho*` to control the folded-null inflation is the deferred V4 work; V1 does not
modify production or the specification to address it.

## V2 contents (Tier-B series-level generator + §2.1 chain validation)

| Module | Purpose |
|---|---|
| `processes.py` | **pure** autocorrelated-process generators (imports no `mechanism`): AR(1)/OU with analytic `tau_int`, coupled `(V, d)` pairs at a target Pearson `r`, and the misspecification stress families (heavy-tailed Student-t innovations, non-AR(1) AR(2), slow-mixing near-unit-root) |
| `generate.py` (extended) | **pure** Tier-B orchestration: `TierBSystemSpec`, `SeriesResidueSpec`, `generate_series_replicates` (synthesises `V(t)` and each `d_i(t)`), `series_digest` |
| `adapters.py` (extended) | the Tier-B production bridge: `recover_residue_effect` / `recover_frames_from_series` run the series through the **production** §2.1 stack (`pearson_both`, `effective_sample_size`, `corrected_standard_error`, `bootstrap_correlation`) and assemble production-schema frames (reusing V1's `build_per_run_frame`); `tierb_hierarchy_config` |
| `tests/test_processes.py`, `tests/test_tierb.py` | analytic anchors, production `tau_int`/`r`/`N_eff`/Fisher-SE/bootstrap recovery, misspecification handling, end-to-end aggregation, determinism |

### What V2 validates (spec §2.1)

Tier B synthesises the **time series** rather than planting the effect field, then
recovers everything through production: `theta = r(V, d_i)` via `pearson_both`;
`N_eff = T / (2 tau_int)` and `sigma^2 = (1 - theta^2)^2 / N_eff` via
`effective_sample_size` + `corrected_standard_error`; the block-bootstrap refinement
via `bootstrap_correlation`. AR(1) anchors the analytic `tau_int = 1/2 (1+phi)/(1-phi)`.

### Observed properties of the current estimator (characterized, not hidden)

* **[KNOWN LIMITATION]** `effective_sample_size` estimates `tau_int` on the **product
  series** `z = (V-mean V)(d-mean d)`, whose autocorrelation differs from the raw
  series'. So the clean AR(1) `tau_int` formula anchors the **raw** series (tested
  directly via `integrated_autocorr_time`), while the product-series `N_eff` is a
  derived quantity V2 characterizes (it shrinks with `phi`).
* **[KNOWN LIMITATION]** For a near-unit-root (`phi -> 1`) series on a **short**
  trajectory, Sokal windowing **under-estimates** `tau_int` (the true value is far
  larger). `test_slow_mixing_iat_is_underestimated_current_estimator` asserts the
  direction honestly; the estimator does not flag this via `status`.
* **[KNOWN LIMITATION]** The block-bootstrap CI **under-covers** under autocorrelation,
  and the shortfall grows with `phi` (~0.95 at `phi=0`, ~0.87 at `phi=0.7`). Tested as
  a characterized trend, not asserted at the nominal 0.95.

These are properties of the current implementation, documented without modifying the
specification or production code.

## V3 contents (closed-form predicted operating characteristics)

| Module | Purpose |
|---|---|
| `predicted.py` | **pure** closed-form Part IV curves (imports no `mechanism`, no generators, no RNG): `lambda_snr`, `rho_from_lambda`/`rho_from_params`, `lambda_star`, `n_eff_from_T`, `sigma2_bar_from_neff`, `predicted_fpr`, `predicted_power`, `predicted_coverage`, `ell_min`, `over_resolution_bound`, and the `predicted_reference_table` helper |
| `tests/test_predicted.py` | analytic-property checks only — limits, monotonicities, identities — **no simulation** |

### What V3 is (and is not)

V3 is the **faithfulness anchor** the roadmap describes: the model-predicted
operating-characteristic curves in closed (Gaussian-approx) form, depending on
**neither the generators (V1/V2) nor the production estimator**. It exists so the
predicted reference can be verified against its own analytic limits *before* the
empirical-vs-predicted comparison (V5) can be tuned to it. V3 performs **no**
empirical Monte Carlo study and **no** empirical-vs-predicted comparison — those are
V5, and additionally depend on the calibrated `rho*` (V4). This ordering is the
roadmap's explicit precedence.

### Closed forms fixed by the spec vs [CHOICE] approximations

The specification fixes several curves in closed form and the rest by properties:

* `rho = lambda/(1+lambda)`, `lambda = beta^2/(tau^2+sigma_bar^2)` — closed **[SPEC]**;
* `N_eff = T/(2 tau_int)`, `sigma_bar^2 ∝ 1/N_eff` — **[SPEC]** (constant exposed);
* `FPR = alpha` by null calibration — **[SPEC]** (V4 sets `rho*`);
* `ell_min` = finest scale with predicted `rho >= rho*` — **[SPEC]** definition;
* over-resolution `~ exp(-c K g^2)`, `g = rho* - rho_ell^true` — **[SPEC]** up to model `c`.

Power and coverage are **not** uniquely determined by the spec (it fixes only their
monotonicities and limits), so they use the **simplest defensible** Gaussian
approximation, marked **[CHOICE]** in the source:

* `predicted_power` models `Theta_bar ~ Normal(beta, (tau^2+sigma_bar^2)/K)` and
  returns `Pr(|Theta_bar| >= sqrt(lambda_star (tau^2+sigma_bar^2)))` — one normal
  CDF, no non-central chi-square machinery. Increasing in `beta^2`, `N_eff`, `K`;
  decreasing in `tau^2`; `->1` as `beta^2 -> inf`.
* `predicted_coverage(K) = nominal + (1-nominal)/K` — the minimal function that is
  `>= nominal` (honest, never anticonservative) for all `K` and converges to
  `nominal` from above as `K -> inf` ("wider-but-honest at small K").

The tests assert only these spec-required properties, never the specific numeric
values of the approximations. The `rho` formula is **not** duplicated: a test pins
`rho_from_params` to `validation.types.RegionTruth.rho`.

## V4 contents (empirical rho* calibration via null surrogates)

| Module | Purpose |
|---|---|
| `surrogates.py` | **pure** null-surrogate generators (imports no `mechanism`): `permute_replicate_labels` (field-level; the canonical scheme), `phase_randomize` / `phase_randomize_pairs` (series-level; preserve the power spectrum, hence `tau_int`/`N_eff`), `power_spectrum` |
| `calibrate.py` | surrogate-based `rho*` calibration reaching production **only** via the `adapters` bridge (lazy import, so `import validation` stays production-free): `calibrate_rho_star` (canonical, ensemble surrogate null), `ensemble_surrogate_null_rho`, `surrogate_null_rho`, `generator_null_rho` (independent validation), `empirical_fpr`, `write_rho_star_yaml` / `load_rho_star_yaml`, `CalibrationResult` |
| `artifacts/rho_star_DENV_K{3,5,10}.yaml` | provenanced calibrated thresholds (data, not code), keyed by (system, K, T, alpha, B, seed) |
| `tests/test_surrogates.py`, `tests/test_calibrate.py` | surrogate properties; out-of-sample FPR control on disjoint beta=0 draws; determinism; artifact round-trip |

### Canonical calibration (surrogate-based) — [SPEC Part V step 11]

`rho*(scale) = upper-alpha quantile of rho_hat under B null surrogates`. The null is
built by drawing an **ensemble** of beta=0 base datasets (from the V1/V2 generators,
reused), replicate-label-permuting each to destroy cross-replicate reproducibility
while preserving the sampling-noise budget, and computing `rho_hat` per scale via the
**production** M4 aggregator (`aggregate_via_production` → `aggregate_reproducibility`).
No threshold is hard-coded — `rho*` is entirely the surrogate-null quantile.

Generator beta=0 draws are used **only** as an *independent* check that the
surrogate-calibrated `rho*` controls FPR out-of-sample (on disjoint seeds); they do
**not** produce the artifact. Surrogate calibration is canonical; generator nulls
validate it.

### [CALIBRATION RESULT] DENV_NS2B_NS3, alpha=0.05

| K | rho*(residue) | rho*(domain) | rho*(chain) | out-of-sample FPR (domain/chain/residue) |
|---|---|---|---|---|
| 3 | 0.533 | 0.777 | 0.862 | 0.043 / 0.048 / 0.042 |
| 5 | 0.587 | 0.834 | 0.899 | 0.040 / 0.048 / 0.051 |
| 10 | 0.554 | 0.824 | 0.898 | 0.035 / 0.028 / 0.043 |

All calibrated `rho*` values sit **well above** the provisional 0.5 — direct evidence
that the folded-energy positivity bias (characterized in V1) makes an un-calibrated
threshold anticonservative, and that calibration is necessary. Out-of-sample FPR is at
or below `alpha = 0.05` (small Monte-Carlo slack), confirming the (Cal) guarantee.

### [KNOWN LIMITATION] Single-base surrogate under-coverage

Surrogates of a *single* base dataset capture only that dataset's within-permutation
variability, which is narrower than the null spread across independent beta=0
realizations; a `rho*` from one base under-covers (out-of-sample FPR > alpha). This is
an observed property of the folded-energy estimator (`A_en` depends only on magnitudes,
which replicate-label permutation preserves). The canonical calibration therefore pools
surrogates across an **ensemble** of null bases, restoring FPR ≤ alpha. Documented
honestly; no change to the specification or production. Production remains
`calibrated: false` — the calibrated `rho*` lives in the validation artifact as data
and is not wired into production in V4.

## Separation boundary (V1/V2/V3/V4, unchanged from V0's principle)

At V0 no validation module imported `mechanism`. V1/V2 open exactly one narrow edge:
only `adapters.py` imports `mechanism`, and only from documented public modules
(`mechanism.config.hierarchy_schema` for the hierarchy translation and the public
`mechanism.statistics` §2.1 functions for Tier-B recovery). The separation tests
enforce that the set of validation non-test modules importing `mechanism` is a subset
of `{adapters.py}`, that only public (non-underscore) names are used, and that they
come only from those two modules; `generate.py`, `processes.py`, `predicted.py`,
`surrogates.py`, and `calibrate.py` stay `mechanism`-free (`calibrate.py` reaches
production only via a lazy `adapters` import) and `import validation` remains
production-free.

## Running the validation tests

The validation suite is intentionally separate from the production suite:

```bash
# from the repository root
python -m pytest validation/tests
```

(The data contract and seeds are pure Python + numpy; no MDAnalysis/POVME needed.)
