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

## Separation boundary (V1, tightened from V0)

At V0 no validation module imported `mechanism`. V1 opens exactly one narrow edge:
only `adapters.py` imports `mechanism`, and only the public
`mechanism.config.hierarchy_schema`. The separation tests enforce that the set of
validation non-test modules importing `mechanism` is a subset of `{adapters.py}` and
that only public (non-underscore) names are used; `generate.py` stays `mechanism`-free
and `import validation` remains production-free.

## Running the validation tests

The validation suite is intentionally separate from the production suite:

```bash
# from the repository root
python -m pytest validation/tests
```

(The data contract and seeds are pure Python + numpy; no MDAnalysis/POVME needed.)
