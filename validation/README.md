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

## Separation boundary (V1/V2, tightened from V0)

At V0 no validation module imported `mechanism`. V1/V2 open exactly one narrow edge:
only `adapters.py` imports `mechanism`, and only from documented public modules
(`mechanism.config.hierarchy_schema` for the hierarchy translation and the public
`mechanism.statistics` §2.1 functions for Tier-B recovery). The separation tests
enforce that the set of validation non-test modules importing `mechanism` is a subset
of `{adapters.py}`, that only public (non-underscore) names are used, and that they
come only from those two modules; `generate.py` and `processes.py` stay
`mechanism`-free and `import validation` remains production-free.

## Running the validation tests

The validation suite is intentionally separate from the production suite:

```bash
# from the repository root
python -m pytest validation/tests
```

(The data contract and seeds are pure Python + numpy; no MDAnalysis/POVME needed.)
