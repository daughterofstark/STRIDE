# VALIDATION_ROADMAP.md
## Incremental roadmap for the validation & benchmarking framework (Phase 2)

This roadmap decomposes the validation study (`VALIDATION_AND_BENCHMARKING.md`,
the deferred Phase-1 item per `IMPLEMENTATION_ROADMAP.md` line 162 and risk R8)
into nine independently testable milestones **V0–V8**, in the spirit of the
M0–M6 implementation roadmap. It follows the mathematical specification exactly,
treats the production `mechanism` package as a frozen black box, and ends with a
publication-grade framework that (per spec Part VII) **empirically confirms the
calibration of ρ\*** and **demonstrates the consequence — that STRIDE refuses the
over-resolved claims single-trajectory practice would emit — on ≥2 systems beyond
DENV**.

### Governing principles (enforced at every milestone)

- **Faithful, not flattering.** Part IV states the operating characteristics are
  *derived, not benchmarked*: the framework measures **empirical vs model-predicted**
  curves. Generators draw from the spec's own model (§2.1–§2.2); no metric or
  assumption is introduced that the model does not already entail. Deliberate
  **misspecification stress tests** prevent the generator from secretly matching
  STRIDE's assumptions.
- **Total separation.** Everything lives in a new top-level `validation/` package
  outside `src/mechanism/`. **No production module is modified.** Production has no
  dependency on `validation/`; deleting `validation/` leaves Phase 1 golden-green.
- **Calibration ≠ validation.** ρ\* is derived on null/surrogate draws, **frozen**,
  then evaluated on **disjoint** signal-bearing draws. No circular reuse.
- **Determinism.** Every draw, surrogate, and resample is seeded and the seed is
  recorded (the Phase-1 R5 discipline), so every number is reproducible.
- **Independently testable, logically complete commits.** Each milestone ships with
  its own tests and is meaningful on its own.

### Milestone overview

| V | Objective | New `validation/` modules | Prod. modified | Headline output |
|---|---|---|---|---|
| V0 | Isolation harness, determinism, data contract | `__init__`, `_seed`, `types`, `cli`(skel) | none | separation + determinism guards |
| V1 | Tier-A field-level ground-truth generator | `generate` | none | planted systems with known (β,τ²,σ̄²,ρ,ℓ\*) |
| V2 | Tier-B series-level generator + §2.1 chain | `processes`, `generate`(ext) | none | autocorrelated series, N_eff/σ² validated |
| V3 | Closed-form predicted operating characteristics | `predicted` | none | model FPR/power/coverage/ℓ_min curves |
| V4 | Empirical ρ\* calibration (surrogate quantile) | `surrogates`, `calibrate` | none | `rho_star.yaml` (provenanced) |
| V5 | Empirical operating characteristics + emp-vs-pred | `metrics` | none | per-cell empirical-vs-predicted tables |
| V6 | Baselines + comparative statistical tests | `baselines`, `stats_tests` | none | over-resolution comparison + p-values |
| V7 | Orchestration, sweep runner, persistence, CLI, reproducibility | `experiments`, `systems/`, `cli` | none\* | reproducible results store + CLI |
| V8 | Publication: figures, manuscript tables, report, artifacts | `figures`, `tables`, `report` | none | `VALIDATION_AND_BENCHMARKING.md` + figures |

\* V7 optionally adds *one additive* production config field to *consume* a
calibrated ρ\*; kept out by default to preserve separation (see V7).

---

## V0 — Foundation: isolation harness, determinism, data contract

1. **Objective.** Stand up the isolated `validation/` package and the guards that
   make every later milestone trustworthy: production stays untouched and uncoupled,
   and all randomness is reproducible. The analog of M0's golden harness.
2. **Mathematical components.** None (infrastructure). Fixes the §2.2 ground-truth
   schema `(β_R, τ_R², σ̄_R², ρ_R, ℓ\*, direction)` as the data contract every later
   milestone populates and checks against.
3. **Modules added.** `validation/__init__.py`; `validation/_seed.py` (seeded RNG
   factory; records seeds); `validation/types.py` (`GroundTruthSystem`, `SimResult`,
   `SweepCell` dataclasses); `validation/cli.py` (skeleton); `validation/README.md`;
   `validation/tests/conftest.py` (validation tests live **outside** production's
   `testpaths = ["tests"]`, so a production run never collects them).
4. **Existing modules modified.** None.
5. **Tests added.** Separation guard: `import mechanism` succeeds and `mechanism`
   (and its submodules) never import `validation`; validation imports only the
   documented public `mechanism` API. Determinism: same seed → byte-identical draws.
   Package smoke and dataclass round-trip.
6. **Expected outputs.** Scaffolding only; a `validation/README.md` stub.
7. **Boundary rationale.** Nothing downstream is trustworthy until isolation and
   determinism are guaranteed; this commit cannot regress production because it adds
   an orthogonal tree and proves the absence of coupling.

## V1 — Tier-A field-level ground-truth generator

1. **Objective.** Generate per-replicate effect fields from the random-effects model
   with fully known ground truth, and plant driver regions (including the I1
   support-permuting structure) at a chosen true scale ℓ\*.
2. **Mathematical components.** §2.2 (Θ_R^(k)=β+γ+e, γ~(0,τ²), e~(0,σ²)); §2.4
   (λ=β²/(τ²+σ̄²), ρ=λ/(1+λ), σ̄²); planting at ℓ\* with null regions elsewhere;
   directional-coherence knob (§2.4/A4) for the signed-vs-mixed path; the I1 planted
   structure (Part III).
3. **Modules added.** `validation/generate.py` (Tier A); extends `types.py`.
4. **Existing modules modified.** None.
5. **Tests added.** Round-trip closure: fields fed to the **production** M4
   `aggregate_reproducibility`/`region_reproducibility` recover the planted ρ within
   tolerance, and the planted scale is reproducible at/above ℓ\* but not below;
   coherence knob produces coherent vs mixed regions; determinism.
6. **Expected outputs.** In-memory `GroundTruthSystem`s and optional serialized
   fixtures for reuse.
7. **Boundary rationale.** The field generator is the substrate for calibration and
   metrics; it is independently testable by round-tripping through the already-validated
   production estimator, so it forms a self-contained, low-risk commit.

## V2 — Tier-B series-level generator + autocorrelation-chain validation

1. **Objective.** Generate autocorrelated V(t)/d_i(t) with known integrated
   autocorrelation time and planted Pearson coupling, and validate the §2.1
   effective-N / sampling-noise chain end-to-end through the production M1/M2 stack;
   include misspecified-noise processes.
2. **Mathematical components.** §2.1 (θ via Pearson; N_eff=T/(2τ_int);
   σ²=(1−θ²)²/N_eff; block bootstrap, M2). AR(1)/OU with analytic τ_int (matches the
   M1 neff test); misspecified (non-AR1, heavy-tailed, slow-mixing) processes for the
   robustness property (R).
3. **Modules added.** `validation/processes.py`; extends `validation/generate.py`
   (Tier B).
4. **Existing modules modified.** None.
5. **Tests added.** Production `neff` recovers the planted τ_int/N_eff within
   tolerance; production correlation recovers the planted r; M2 bootstrap CI covers
   the known r at nominal rate; misspecified series are handled/flagged, not silently
   trusted.
6. **Expected outputs.** Synthetic series fixtures + recovered-parameter tables.
7. **Boundary rationale.** Tier B exercises a *different* spec layer (§2.1 sampling
   noise) than Tier A (§2.2 variance components) and is computationally heavier;
   isolating it keeps V1 lean and makes the autocorrelation-chain validation its own
   logically complete commit — and it is exactly the layer that distinguishes STRIDE
   from naive averaging.

## V3 — Closed-form predicted operating characteristics

1. **Objective.** Implement the model-predicted operating-characteristic curves in
   closed (Gaussian-approx) form, so V5 can perform the Part IV **empirical-vs-predicted**
   comparison against an auditable reference.
2. **Mathematical components.** Part IV in full: λ and ρ=λ/(1+λ); FPR=α by
   construction; power as a non-central function of (β², τ², N_eff, K), increasing in
   β²/N_eff/K and decreasing in τ²; coverage of the β̂ interval; the minimum resolvable
   scale ℓ_min(K,T,τ_int,g); the over-resolution bound ≈ exp(−cKg²).
3. **Modules added.** `validation/predicted.py`.
4. **Existing modules modified.** None.
5. **Tests added.** Analytic-property checks: ρ→1 as λ→∞; monotonicities of power;
   ℓ_min non-increasing in K and T; predicted FPR=α; over-resolution probability
   decreasing in K and gap g. No simulation needed — pure math against its own limits.
6. **Expected outputs.** Predicted-curve functions and reference tables.
7. **Boundary rationale.** These curves are the faithfulness anchor (Part IV "derived,
   not benchmarked"). They depend on neither generators nor production, so isolating
   them lets the predicted reference be verified against analytic properties *before*
   any empirical comparison can be tuned to it.

## V4 — Empirical ρ\* calibration (surrogate quantile)

1. **Objective.** Implement the spec's actual calibration — the deferred Phase-1
   methods contribution (R8) — and enforce the calibration/validation split.
2. **Mathematical components.** Part V step 11: ρ\* = upper-α quantile of ρ̂ under B
   null surrogates, with both surrogate schemes (replicate-label permutation;
   phase randomization of Tier-B series, preserving the autocorrelation spectrum so
   N_eff is correct under the null); the (Cal) property Pr(ℓ̂\*≤ℓ | ρ_ℓ<ρ\*) ≤ α.
3. **Modules added.** `validation/surrogates.py`, `validation/calibrate.py`.
4. **Existing modules modified.** None (computes ρ̂ via the production
   `run_aggregation`/`region_reproducibility` API).
5. **Tests added.** Out-of-sample FPR control: ρ\* calibrated on null/surrogate draws
   yields empirical FPR ≤ α on **disjoint** β=0 draws (the (Cal) guarantee); both
   surrogate schemes yield valid null distributions; the train/test seed split is
   enforced (a test asserting disjoint seed streams); seeded determinism (R5).
6. **Expected outputs.** A provenanced `rho_star.yaml` artifact keyed by
   (system, K, T, α, B, seed) — the empirical replacement for the provisional value,
   delivered as data, not code.
7. **Boundary rationale.** Calibration is the core novelty (Part VII) and must precede
   any metric evaluated *at* ρ\* (V5). It depends on the generators (V1/V2) and is
   independently testable via the FPR-control property. "We can now set ρ\* honestly"
   is a logically complete commit.

## V5 — Empirical operating characteristics + empirical-vs-predicted check

1. **Objective.** Measure the empirical operating characteristics on simulated
   ensembles at the calibrated ρ\* and compare them to V3's predicted curves (the
   Part IV calibration check); add the practitioner summary statistics.
2. **Mathematical components.** Part IV (empirical FPR/power/coverage/ℓ_min); the
   over-resolution bound; the (C) consistency property (estimates→truth as K,T→∞);
   the I2 (upward-closed passable set) and I3 (standardization invariance) checks on
   simulated systems. Precision/recall/ROC are computed as the aggregate summary of
   the scale-selection task (flagged as reporting, mapping onto the spec curves, not a
   new estimand).
3. **Modules added.** `validation/metrics.py`.
4. **Existing modules modified.** None.
5. **Tests added.** Empirical curves match predicted within tolerance on seeded
   ensembles; consistency holds as K,T grow; I2/I3 invariants hold; over-resolution
   decays with K and gap; ROC/AUC computed correctly on constructed cases.
6. **Expected outputs.** Per-(K,T,τ²,β²) metric tables pairing empirical with
   predicted values, plus the ℓ_min grid.
7. **Boundary rationale.** Metrics require the generators (V1/V2), the calibrated ρ\*
   (V4), and the predicted reference (V3); this is where the central faithfulness
   check lands, and it is independently testable against V3's closed form.

## V6 — Baselines + comparative statistical tests

1. **Objective.** Implement the Part VI baselines and the paired statistical tests,
   and deliver the Part VII consequence: STRIDE refuses over-resolution that
   single-trajectory and naive practice emit.
2. **Mathematical components.** Part VI baselines as special cases/relatives —
   single-trajectory argmax_i|θ_i^(1)| (unbounded reproducibility uncertainty),
   ensemble averaging (mean±SD, SD/√K, ℓ=0 special case), residue-level ranking, and
   optionally IDR (the named closest relative) and a fixed-resolution G-theory
   coefficient. Part IV coverage bullet (honest interval vs anticonservative SD/√K).
   Part VII over-resolution consequence.
3. **Modules added.** `validation/baselines.py`, `validation/stats_tests.py`.
4. **Existing modules modified.** None.
5. **Tests added.** Each baseline reproduces its known degenerate behaviour
   (single-traj always residue-scale/over-resolves; naive SD/√K under-covers at K=3);
   McNemar / DeLong / Wilcoxon / paired-bootstrap return correct results on constructed
   inputs; Benjamini–Hochberg control across the grid; the headline paired result
   (STRIDE over-resolution rate < baselines) on planted nulls.
6. **Expected outputs.** Method-comparison tables and pre-registered test statistics /
   p-values per claim.
7. **Boundary rationale.** Baselines and their comparisons are the comparative layer,
   separable from the within-STRIDE characteristics (V5). This milestone delivers the
   Part VII demonstration and is testable via known baseline degeneracies and
   constructed test cases.

## V7 — Orchestration, sweep runner, persistence, CLI, reproducibility

1. **Objective.** The final **engineering** milestone: drive the full sweep over the
   (K, T, τ², β², system) grid on ≥2 distinct synthetic systems beyond DENV, persist
   every result deterministically, and expose a reproducible command-line runner. No
   figures, no manuscript assembly (those are V8). The integration backbone (analog of
   M6) onto which V8 attaches.
2. **Mathematical components.** No new mathematics — orchestration of Parts IV–VII
   already implemented in V1–V6; the ℓ_min grid and the hierarchy-sensitivity sweep
   (risk R6); enforcement of the Part VII "≥2 systems beyond DENV" coverage.
3. **Modules added.** `validation/experiments.py` (sweep grid, cell execution,
   result persistence with seed/provenance), `validation/systems/` (≥2 system
   definitions with distinct hierarchies), `validation/cli.py` (complete:
   `python -m validation run …`, `calibrate …`, `sweep …`).
4. **Existing modules modified.** None by default. *Optional, flagged:* one additive
   production config field to let production *consume* a calibrated ρ\* with provenance
   (the `mechanism.json` flag flips to calibrated only on explicit opt-in). Kept out of
   scope unless requested, to preserve total separation.
5. **Tests added.** End-to-end smoke on a tiny deterministic grid (fast, seeded);
   result-store round-trip (write→read identical); CLI re-run reproducibility (identical
   numbers across runs); an assertion that ≥2 non-DENV systems are exercised; provenance
   (seed, B, ρ\*, versions) recorded in every result record.
6. **Expected outputs.** A machine-readable results store (per-cell records with
   provenance), the frozen `rho_star.yaml`, and a reproducible CLI. **No figures, no
   report.**
7. **Boundary rationale.** Separating the engineering backbone (deterministic
   execution, persistence, CLI) from publication output keeps the reproducibility layer
   independently verifiable and stable; V8's figures and prose then attach to a frozen,
   tested results store rather than recomputing. This mirrors the M-phase discipline of
   isolating wiring from presentation.

## V8 — Publication: figures, manuscript tables, report, artifacts

1. **Objective.** The **publication** milestone: turn the frozen V7 results store into
   publication-quality figures, manuscript tables, the `VALIDATION_AND_BENCHMARKING.md`
   report, and the final reproducibility package — the artifacts that clear the spec's
   Part VII bar.
2. **Mathematical components.** No new mathematics — presentation of the V3 predicted
   vs V5 empirical curves, the V4 calibration, the V6 comparisons, and the ℓ_min
   headline (Parts IV–VII).
3. **Modules added.** `validation/figures.py` (all publication figures),
   `validation/tables.py` (manuscript tables), `validation/report.py` (assembles the
   report), and the generated `VALIDATION_AND_BENCHMARKING.md`.
4. **Existing modules modified.** None.
5. **Tests added.** Figures render from the persisted results store (no recomputation);
   tables match the underlying records; the report assembles and includes the required
   sections (calibration, empirical-vs-predicted, baseline comparison, ℓ_min,
   hierarchy sensitivity, ≥2 systems); the final reproducibility package re-builds from
   the store deterministically.
6. **Expected outputs.** The figure set (calibration curve, empirical-vs-predicted
   FPR/power, coverage, the ℓ_min heatmap, the over-resolution comparison, ROC/PR,
   example Π profiles, hierarchy-sensitivity panel), manuscript tables, the
   `VALIDATION_AND_BENCHMARKING.md` report, and the packaged artifacts.
7. **Boundary rationale.** Publication output reads a frozen, tested results store and
   produces no new numbers, so figures/prose can iterate without touching the validated
   computation — the cleanest possible separation between *what was measured* (V0–V7)
   and *how it is presented* (V8), and the natural final commit.

---

## Validation risk analysis (preserving faithfulness & separation)

| # | Risk | Where | Mitigation |
|---|---|---|---|
| VR1 | **Self-flattery**: generator silently matches STRIDE's Gaussian/AR assumptions, inflating performance | V1, V2, V5 | Generate from the spec's own model *and* include misspecified-noise stress processes; report empirical **vs predicted** (Part IV), not absolute scores |
| VR2 | **Calibration circularity**: ρ\* set and evaluated on the same draws | V4, V5 | Frozen train/test seed split; FPR control demonstrated **out-of-sample**; split asserted in tests |
| VR3 | **Production coupling / regression** | all | `validation/` is a separate top-level package; **zero** production edits; V0 separation guard test; production has no reverse dependency |
| VR4 | **Nondeterminism** ⇒ irreproducible curves | all | Seed every draw/surrogate/resample; record seeds (R5); CLI re-run reproducibility test (V7) |
| VR5 | **Provisional ρ\* mistaken for production default** | V4, V7 | ρ\* delivered as a provenanced data artifact; production stays `calibrated:false` unless a user explicitly opts in (R8) |
| VR6 | **Synthetic ≠ biological generality** | V7 | State scope honestly: synthetic ground truth is the correct instrument for operating characteristics (ℓ\* is unknowable on real data); biological generality is a further, separate step |
| VR7 | **Test/method chosen post hoc** | V6 | Pre-register the statistical test for each claim; BH correction across the grid |

## End state

After V8 the repository carries the complete Phase-1 production framework (M0–M6,
unchanged) **plus** a fully separate `validation/` framework that: generates
spec-faithful synthetic ground truth (field- and series-level), empirically
calibrates ρ\* by the Part V surrogate quantile with an honest train/test split,
confirms the Part IV operating characteristics match their closed-form predictions,
benchmarks STRIDE against the Part VI baselines, and demonstrates the Part VII
consequence — refusing over-resolution that single-trajectory practice emits — on
≥2 systems beyond DENV. That is precisely the empirical evidence the spec's Part VII
names as the bar between "good statistics, weak novelty" and "publishable methodology."
