# STRIDE

**STRIDE** infers mechanistic claims from replicate molecular-dynamics (MD) simulations
**at — and only at — the spatial resolution the replicate evidence supports.**

Standard single-trajectory practice reports `argmax_i |θ_i|` — the residue with the
largest effect in one trajectory — as "the mechanism." That is a residue-resolution claim
with *unbounded reproducibility uncertainty*: nothing guarantees the same residue would
appear in an independent replicate. STRIDE instead asks a well-posed question — *at what
spatial scale is a mechanistic effect field reproducible across independent replicates,
relative to autocorrelation-corrected sampling noise?* — estimates a **calibrated
reproducibility threshold**, and **refuses to over-resolve** when the data do not support
a fine-grained claim.

> **Core idea.** For each region `R` at hierarchy scale `ℓ`, STRIDE estimates a
> reproducibility coefficient `ρ_R = β_R² / (β_R² + τ_R² + σ̄_R²) ∈ [0, 1]` — the fraction
> of effect variance that is reproducible signal (`β` = reproducible effect, `τ²` =
> between-replicate variance, `σ̄²` = autocorrelation-corrected within-replicate variance).
> A region passes at scale `ℓ` if `ρ̂_R ≥ ρ*`, where `ρ*` is an empirically **calibrated**
> threshold with a controlled false-resolution rate. The gate returns `ℓ̂*`, the finest
> scale at which the mechanistic claim is reproducible — coarsening (the safe error) when
> the evidence is weak.

---

## Features

### Production framework (`mechanism`)
- Reproducibility-gated mechanistic inference from replicate MD trajectories.
- Per-residue effect (`θ`) estimation with **autocorrelation-corrected** uncertainty
  (effective sample size, block bootstrap).
- A residue → domain → chain **hierarchy** model with canonical numbering.
- **Variance-component** decomposition (`β`, `τ²`, `σ̄²`) and the reproducibility
  coefficient `ρ`.
- A per-locus **resolution profile `Π`**, a calibrated **gate**, and emitted
  **mechanisms** at the gated scale `ℓ̂*`.
- Byte-for-byte reproduction of the original "v5" analysis (a golden-tested,
  behaviour-preserving foundation).

### Validation framework (`validation/`)
- **Synthetic ground-truth generators** (field-level and autocorrelated series-level).
- **Closed-form predicted** operating characteristics (FPR, power, coverage, `ℓ_min`,
  over-resolution bound).
- **Empirical `ρ*` calibration** via surrogate-null quantiles.
- **Empirical-vs-predicted** operating-characteristic checks (FPR, power, coverage,
  consistency, the I2/I3 invariances).
- **Baselines and comparative tests** (single-trajectory, naive ensemble, residue
  ranking, G-theory; McNemar / Wilcoxon / DeLong / paired bootstrap / Benjamini–Hochberg).
- A reproducible **sweep runner**, deterministic **persistence**, and a **CLI**.
- A **publication layer**: figures, manuscript tables, and the generated
  `VALIDATION_AND_BENCHMARKING.md` report.

---

## Mathematical foundations

STRIDE estimates `ℓ*`, the finest spatial scale at which a mechanistic effect field is
reproducible across independent replicates, together with its scale-indexed profile `Π`
and a calibrated gate `ℓ̂*`. The reproducibility coefficient `ρ` is a
reliability/generalizability coefficient (the fraction of effect variance that is
reproducible signal); the threshold `ρ*` is calibrated as the upper-`α` quantile of `ρ̂`
under a surrogate null, giving a controlled false-resolution rate. The estimator obeys
structural invariances — region reproducibility dominates residue reproducibility when a
signal permutes within a region (I1), the passable set is upward-closed (I2), the gate is
standardization-invariant (I3), and it is monotone in `ρ*` (I4) — and is consistent as
`K, T → ∞`.

The full method, its derivations, and its operating-characteristic theory are in
[`MATHEMATICAL_SPECIFICATION.md`](MATHEMATICAL_SPECIFICATION.md). This README does not
reproduce the specification.

---

## Repository structure

```
src/mechanism/     Production package (installed as `mechanism`). Frozen scientific instrument.
validation/        Validation & benchmarking framework (Phase 2). Not installed. Imports
                   mechanism read-only through a single bridge; mechanism never imports it.
tests/             Production tests (v5-parity + golden regression harness).
reference/         Original v5 source, kept for source-identity tests.
scripts/           Golden-fixture capture.
configs/           Example hierarchy configuration (denv_hierarchy.yaml).
.github/           CI workflows.
MATHEMATICAL_SPECIFICATION.md   The method (ground truth).
IMPLEMENTATION_ROADMAP.md       Production roadmap (M0–M6).
VALIDATION_ROADMAP.md           Validation roadmap (V0–V8).
VALIDATION_AND_BENCHMARKING.md  Generated validation report (V8).
```

---

## Installation

Requirements: **Python ≥ 3.10**. Core dependencies: numpy, pandas, scipy, matplotlib,
MDAnalysis, PyYAML.

```bash
pip install -e .            # core (includes MDAnalysis) — to run the production pipeline
pip install -e ".[dev]"     # + pytest, pillow (test suite)
pip install -e ".[msa]"     # + biopython (optional MSA features)
```

The `validation/` package is intentionally **not** installed (it lives outside `src/`).
Run it in place from the repository root.

---

## Quick start

Analyze a set of replicate trajectories and obtain gated mechanisms:

```python
from mechanism import Config, run_pipeline

# `base_dir` points to a directory of replicate ("triplicate") trajectories laid out
# as the pipeline expects; see configs/denv_hierarchy.yaml for the hierarchy definition.
run_pipeline(Config(base_dir="/path/to/TRIPLICATES"))
```

or from the command line (flags mirror the original v5 pipeline):

```bash
mechanism /path/to/TRIPLICATES --stride 20 --frame-ps 10
```

Outputs include the v5-parity analysis tables/figures plus a **mechanism report**: for
each region, the gated scale `ℓ̂*`, the reproducibility coefficient `ρ̂`, the signed
effect `β̂` with a confidence interval, and a direction. A mechanism emitted at *domain*
scale means the replicate evidence supports a domain-level claim — STRIDE has refused to
over-resolve to individual residues.

---

## Running the validation framework

The validation study is **already complete**; its results are frozen in the repository.
These commands are primarily for **verification** or **extension**. Run from the repo root.

```bash
# Calibrate a reproducibility threshold rho* (explicit step; writes a frozen artifact)
python -m validation calibrate --system DENV_NS2B_NS3 --K 5

# Run the operating-characteristic sweep over the (system, K, T, tau2, beta2) grid
# (loads calibrated rho*; never calibrates implicitly; deterministic & reproducible)
python -m validation sweep \
  --systems DENV_NS2B_NS3 two_level_single_chain three_level_two_chain \
  --K 3 5 10 --beta2 0.16 0.36 1.0 --n-eval 50 \
  --out validation/artifacts/sweep_results.jsonl

# Build the publication package (figures + tables + report) from the frozen store
python -m validation report --out validation/artifacts/publication
```

Run the validation test suite with `python -m pytest validation/tests` (269 tests, ~3 min).

---

## Outputs

**Reproducibility inputs (frozen; regenerate only deliberately):**
- `validation/artifacts/rho_star_<system>_K<K>.yaml` — calibrated thresholds `ρ*` by
  scale, with confidence intervals and provenance (B, α, seed, surrogate).
- `validation/artifacts/metrics_DENV.yaml` — empirical-vs-predicted operating points and
  the `ℓ_min` grid.
- `validation/artifacts/method_comparison_DENV.yaml` — over-resolution comparison, naive
  coverage, and Benjamini–Hochberg results.
- `validation/artifacts/sweep_results.jsonl` (+ `_manifest.json`) — the frozen per-cell
  results store with a SHA-256 `results_digest`.

**Publication outputs (regenerable from the frozen inputs):**
- `validation/artifacts/publication/figures/*.png` — calibration curve, empirical-vs-
  predicted FPR/power, coverage, `ℓ_min` heatmap, over-resolution comparison,
  hierarchy-sensitivity panel, profile schematic.
- `validation/artifacts/publication/tables/*.md` — five manuscript tables.
- `VALIDATION_AND_BENCHMARKING.md` — the assembled validation report.

---

## Validation status

**Production**
- ✓ M0 (v5-parity foundation) · ✓ M1 · ✓ M2 · ✓ M3 · ✓ M4 · ✓ M5 · ✓ M6

**Validation**
- ✓ V0 · ✓ V1 · ✓ V2 · ✓ V3 · ✓ V4 · ✓ V5 · ✓ V6 · ✓ V7 · ✓ V8

**Regression (current):**
- Production: **133 passed, 2 skipped** (135 collected).
- Validation: **269 passed**.
- `src/mechanism/_legacy.py` md5 = `2005822cc5b2a8cc3a6b5e58425043eb` (invariant).
- Production version `0.6.0+m6`; validation version `0.9.0+v8`.

**Headline empirical results** (synthetic systems, at the calibrated `ρ*`): false-positive
rate controlled at `α`; empirical power tracks the closed-form prediction at
moderate/high SNR; and — the central consequence — STRIDE's over-resolution rate on
planted nulls is `0.000` versus `1.000` (single-trajectory) and `0.48–0.73` (naive
ensemble), demonstrated on the DENV anchor and **two additional systems with distinct
hierarchies** (paired McNemar `p < 6e-17`; all comparisons survive Benjamini–Hochberg).

---

## Reproducibility

- **Deterministic seeds.** All randomness flows through `validation/_seed.py`, with
  disjoint seed streams for calibration and evaluation.
- **Frozen artifacts.** Calibration thresholds and the sweep results store are persisted
  with provenance; a SHA-256 digest pins the store
  (`1993d3e731d067c9d1333293489851b3bdf13d6902c014c34ea517dd7af88c9f`).
- **Regenerate everything.** A sweep re-run with the same grid/seed yields a
  **byte-identical** store; the report rebuilds identically:
  ```bash
  python -m validation report --out /tmp/verify
  ```
- **Separation is tested.** `mechanism` never imports `validation`; only
  `validation/adapters.py` imports `mechanism`; `import validation` is mechanism-free.

---

## Documentation

- [`MASTER_HANDOFF.md`](MASTER_HANDOFF.md) — complete technical history and rationale.
- [`CURRENT_PROJECT_STATE.md`](CURRENT_PROJECT_STATE.md) — exact state of the repository.
- [`USER_GUIDE.md`](USER_GUIDE.md) — how to install, run, calibrate, sweep, and report.
- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) — architecture, boundaries, and how to extend.
- [`SCIENTIFIC_VALIDATION.md`](SCIENTIFIC_VALIDATION.md) — what was validated and measured.
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — what to do next (including real-data application).
- [`MATHEMATICAL_SPECIFICATION.md`](MATHEMATICAL_SPECIFICATION.md) — the method.
- [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) · [`VALIDATION_ROADMAP.md`](VALIDATION_ROADMAP.md) — roadmaps.

---

## Current status

The implementation (M0–M6) and the validation framework (V0–V8) are **complete**. The
method is calibrated and its operating characteristics are characterized on synthetic
systems with known ground truth, including ≥ 2 systems beyond DENV.

**What remains** is scientific application and open research, not framework construction:
- **Application to biological datasets** — running STRIDE on real DENV1–4 replicate
  trajectories under the claimed operating range (`K ≥ 5`) and a calibrated `ρ*`. This is
  the next required step; see [`NEXT_STEPS.md`](NEXT_STEPS.md).
- **Future research** — reconciling the low-SNR power model, real-null calibration,
  alternative effect fields, and a biological-generality study.
- **Optional extensions** — a full IDR baseline, richer reporting (full ROC/PR curves,
  live `Π` profiles), and larger sweeps.

**Scope, honestly stated:** the validation establishes operating characteristics on
*synthetic* systems. Synthetic ground truth is the correct instrument for this — `ℓ*` is
unknowable on real data — but biological generality is a separate, later step and is not
claimed here.

---

## Citation

A methods paper is in preparation. Until then, please cite the repository:

```bibtex
@software{stride,
  title  = {STRIDE: reproducibility-gated mechanistic inference from replicate
            molecular-dynamics simulations},
  author = {The STRIDE authors},
  year   = {2026},
  note   = {Version: production 0.6.0+m6, validation 0.9.0+v8},
  url    = {<repository URL>}
}
```

*(Placeholder — update with authors, title, venue, and DOI on publication.)*

---

## License

MIT License © 2026 the authors. See [`LICENSE`](LICENSE) for the full text.
