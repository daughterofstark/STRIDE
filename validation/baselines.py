"""Part VI baseline estimators for the comparative layer (milestone V6).

**Pure** module: imports nothing from ``mechanism`` and no RNG of its own beyond what
callers pass in. Each baseline is one of the existing methods the specification (Part
VI) positions STRIDE against, implemented as a *special case or relative* so the
comparison is apples-to-apples on the same per-run effect frames:

* **single-trajectory** ``argmax_i |theta_i^(1)|`` — current single-run practice; uses
  only one replicate, always emits a **residue-scale** claim, and has *unbounded*
  reproducibility uncertainty (``tau^2`` is unidentifiable at ``K = 1``). [SPEC VI]
* **naive ensemble averaging** — ``beta_hat = mean_k theta^(k)`` at fixed residue scale
  with an anticonservative ``SD / sqrt(K)`` interval; STRIDE restricted to ``ell = 0``
  with ``tau^2`` folded into ``sigma^2``; cannot detect non-stationarity or select a
  scale. [SPEC VI]
* **residue-level ranking** — rank residues by mean ``|theta|``; a fixed-resolution
  "which items" method (the IDR family's question), no scale selection. [SPEC VI]
* **G-theory coefficient** — a single fixed-resolution generalizability/reliability
  coefficient ``E rho^2 = var_obj / (var_obj + var_resid / K)``; "one coefficient at
  one resolution" (Part VI hierarchical/G-theory relative). [SPEC VI]

[CHOICE] Optional baseline: the roadmap marks **IDR** and **G-theory** as optional. We
implement **G-theory** and **defer IDR**. Rationale: the G-theory coefficient is a
closed-form reliability ratio over the same variance components STRIDE already works
with (a few lines, no new dependencies, and structurally a fixed-resolution special
case of ``rho``), so it aligns directly with the existing architecture. IDR
(Li-Brown-Huang-Bickel 2011) requires fitting a two-component **copula mixture by EM**
over ranked replicate pairs — substantial new machinery, classically limited to two
replicates, and answering an orthogonal "which items" question rather than the
scale-selection question. It is documented here as the *named closest relative* and
represented in the comparison by residue-level ranking (its fixed-resolution,
which-items core), with the full copula-mixture EM deferred as out-of-scope for V6.

Over-resolution (the Part VII consequence)
------------------------------------------
The comparative headline is that STRIDE *refuses* over-resolution that these baselines
emit. Each baseline exposes an ``*_over_resolves`` predicate: whether, on a given
dataset whose true reproducible scale is coarser than residue, the baseline
nonetheless emits a residue-scale claim. STRIDE's own over-resolution rate is measured
by the existing V5 ``validation.metrics.empirical_over_resolution_rate`` (reused, not
duplicated); this module only supplies the *baseline* side of the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


_EFFECT_COL = "r"
_ABS_COL = "abs_r"
_RESID_COL = "canon_resid"


def _effect_matrix(frames: Sequence, effect_col: str = _EFFECT_COL) -> np.ndarray:
    """Stack K per-run frames into a ``(K, n_residues)`` effect matrix."""
    if len(frames) == 0:
        return np.empty((0, 0))
    n = len(frames[0])
    for f in frames:
        if len(f) != n:
            raise ValueError("all frames must share residue order and length")
    return np.stack([f[effect_col].to_numpy(dtype=float) for f in frames])


# ── single-trajectory argmax (K=1 practice) ──────────────────────────────────
@dataclass(frozen=True)
class SingleTrajClaim:
    """A single-trajectory claim: always residue-scale (the degenerate baseline)."""

    scale_level: str          # always "residue"
    residue_index: int        # argmax position in the effect vector
    canon_resid: int
    magnitude: float


def single_trajectory_claim(frames: Sequence, *, replicate: int = 0,
                            effect_col: str = _EFFECT_COL,
                            resid_col: str = _RESID_COL) -> SingleTrajClaim:
    """``argmax_i |theta_i^(1)|`` from one replicate — always a residue-scale claim.

    Current single-run practice: pick the residue with the largest |effect| in a
    single trajectory. It is *always* a residue-scale claim (it has no notion of a
    coarser scale and no cross-replicate reproducibility check), which is exactly the
    over-resolution the framework is designed to refuse. [SPEC VI]
    """
    f = frames[replicate]
    abs_vals = np.abs(f[effect_col].to_numpy(dtype=float))
    j = int(np.argmax(abs_vals))
    canon = int(f[resid_col].to_numpy()[j])
    return SingleTrajClaim(scale_level="residue", residue_index=j,
                           canon_resid=canon, magnitude=float(abs_vals[j]))


def single_trajectory_over_resolves(frames: Sequence, *, true_scale_level: str,
                                    **kw) -> bool:
    """True iff the single-trajectory claim is finer than truth.

    Since the claim is *always* residue-scale, this over-resolves whenever the true
    reproducible scale is coarser than residue — i.e. for every non-residue truth.
    """
    _ = single_trajectory_claim(frames, **kw)
    return true_scale_level != "residue"


# ── naive ensemble averaging (ell=0, SD/sqrt(K)) ─────────────────────────────
@dataclass(frozen=True)
class NaiveEnsembleClaim:
    """Naive residue-scale ensemble average with an SD/sqrt(K) interval."""

    mean: np.ndarray          # per-residue mean effect
    sd: np.ndarray            # per-residue SD across replicates
    se: np.ndarray            # SD / sqrt(K) (anticonservative)
    significant: np.ndarray   # |mean| >= z * se  (per residue)
    any_significant: bool
    K: int


def naive_ensemble_claim(frames: Sequence, *, z: float = 1.96,
                         effect_col: str = _EFFECT_COL) -> NaiveEnsembleClaim:
    """Naive ensemble averaging at fixed residue scale with an ``SD/sqrt(K)`` CI.

    ``beta_hat = mean_k theta^(k)`` per residue; a residue is "significant" if
    ``|mean| >= z * SD / sqrt(K)``. This is STRIDE restricted to ``ell = 0`` with
    ``tau^2`` folded into ``sigma^2`` — it discards the scale dimension and the
    between/within split, and its ``SD/sqrt(K)`` interval is anticonservative at small
    ``K`` (Part IV coverage bullet). [SPEC VI]
    """
    arr = _effect_matrix(frames, effect_col)
    K = arr.shape[0]
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if K > 1 else np.full(arr.shape[1], np.nan)
    se = sd / np.sqrt(K)
    with np.errstate(invalid="ignore"):
        sig = np.abs(mean) >= z * se
    return NaiveEnsembleClaim(mean=mean, sd=sd, se=se, significant=sig,
                              any_significant=bool(np.any(sig)), K=K)


def naive_ensemble_over_resolves(frames: Sequence, *, true_scale_level: str,
                                 z: float = 1.96, effect_col: str = _EFFECT_COL,
                                 driver_indices: Optional[Sequence[int]] = None) -> bool:
    """True iff naive averaging emits a residue-scale claim finer than truth.

    A residue-scale claim is emitted when any residue is individually significant. If
    the true reproducible scale is coarser than residue, that is over-resolution. When
    ``driver_indices`` is given, only significance *outside* the driver support counts
    as a false residue-scale claim on a null region; by default any residue
    significance on a coarser-truth system counts.
    """
    claim = naive_ensemble_claim(frames, z=z, effect_col=effect_col)
    if true_scale_level == "residue":
        return False
    if driver_indices is None:
        return bool(claim.any_significant)
    mask = np.ones(claim.significant.shape[0], dtype=bool)
    mask[list(driver_indices)] = False
    return bool(np.any(claim.significant[mask]))


def naive_coverage(target: np.ndarray, means: np.ndarray, ses: np.ndarray,
                   *, z: float = 1.96) -> float:
    """Empirical coverage of naive ``mean +/- z*SD/sqrt(K)`` intervals about ``target``.

    ``means``/``ses`` are ``(n_trials, n_residues)``; ``target`` is the per-residue
    value the interval is meant to cover. Returns the fraction of (trial, residue)
    intervals that contain the target. Anticonservative (< nominal) at small ``K`` —
    the Part IV coverage bullet, quantified in the V6 report.
    """
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    lo = means - z * ses
    hi = means + z * ses
    inside = (lo <= target[None, :]) & (target[None, :] <= hi)
    finite = np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return float("nan")
    return float(inside[finite].mean())


# ── residue-level ranking (IDR's which-items core, fixed resolution) ─────────
def residue_ranking_claim(frames: Sequence, *, effect_col: str = _ABS_COL,
                          resid_col: str = _RESID_COL) -> list:
    """Rank residues by mean ``|theta|`` across replicates (fixed residue resolution).

    Returns a list of ``(canon_resid, mean_abs_effect)`` sorted descending. This is the
    fixed-resolution "which items are reproducible" question (the IDR family's core),
    with no scale selection — represented here as the named IDR relative. [SPEC VI]
    """
    arr = _effect_matrix(frames, effect_col)
    mean_abs = arr.mean(axis=0)
    canon = frames[0][resid_col].to_numpy()
    order = np.argsort(-mean_abs)
    return [(int(canon[j]), float(mean_abs[j])) for j in order]


# ── G-theory coefficient (fixed-resolution reliability) ──────────────────────
def gtheory_coefficient(frames: Sequence, *, indices: Optional[Sequence[int]] = None,
                        effect_col: str = _EFFECT_COL) -> float:
    """Generalizability (reliability) coefficient at a fixed resolution.

    ``E rho^2 = var_obj / (var_obj + var_resid / K)`` from a one-facet
    (objects x replicates) decomposition, where "objects" are the residues in
    ``indices`` (default: all). This is the classical G-theory / Cronbach-era
    reliability coefficient: *one* coefficient at *one* resolution, with no profile and
    no gate — the fixed-resolution relative of ``rho`` (Part VI). Returns a value in
    ``[0, 1]``.

    Interpretation note: G reflects the reliability with which the measurement
    distinguishes *residues from each other* across the replicate facet — it is high
    when residues carry *distinct* magnitudes that are stable across replicates, and
    near 0 when residues are indistinguishable (e.g. a uniform distributed carrier that
    gives every residue the same expected magnitude) even if each residue is
    individually reproducible. This is a fixed-resolution "which items differ" view,
    orthogonal to STRIDE's scale-selection question — exactly the Part VI distinction.
    """
    arr = _effect_matrix(frames, effect_col)
    if indices is not None:
        arr = arr[:, list(indices)]
    K, n = arr.shape
    if K < 2 or n < 1:
        return float("nan")
    obj_means = arr.mean(axis=0)
    grand = arr.mean()
    ms_obj = (K * np.sum((obj_means - grand) ** 2) / (n - 1)) if n > 1 else 0.0
    resid = arr - obj_means[None, :]
    ms_resid = np.sum(resid ** 2) / ((K - 1) * n)
    var_obj = max((ms_obj - ms_resid) / K, 0.0)
    denom = var_obj + ms_resid / K
    if denom <= 0:
        return 0.0
    return float(np.clip(var_obj / denom, 0.0, 1.0))


# ── method-comparison pipeline (baseline over-resolution vs STRIDE) ──────────
def baseline_over_resolution_rates(
    make_frames, *, seeds, true_scale_level: str, z: float = 1.96,
    driver_indices=None,
) -> dict:
    """Per-baseline over-resolution rate on a seeded ensemble (the comparison inputs).

    ``make_frames(seed) -> list[frame]`` builds one system's per-run frames whose true
    reproducible scale is ``true_scale_level``. Returns per-baseline over-resolution
    rates and the per-seed indicator vectors (for paired tests against STRIDE, which is
    measured separately via the reused V5 metric). Pure: no ``mechanism`` here.
    """
    single, naive = [], []
    for s in seeds:
        frames = make_frames(s)
        single.append(int(single_trajectory_over_resolves(
            frames, true_scale_level=true_scale_level)))
        naive.append(int(naive_ensemble_over_resolves(
            frames, true_scale_level=true_scale_level, z=z,
            driver_indices=driver_indices)))
    single = np.asarray(single)
    naive = np.asarray(naive)
    return {
        "single_trajectory": {
            "rate": float(single.mean()) if single.size else float("nan"),
            "indicators": single,
        },
        "naive_ensemble": {
            "rate": float(naive.mean()) if naive.size else float("nan"),
            "indicators": naive,
        },
    }


def build_method_comparison(
    make_null_frames, make_driver_frames, hierarchy_config, *,
    seeds, rho_star: float, true_scale_level: str, driver_region_substr: str,
    driver_indices=None, protein: str = "", z: float = 1.96, boot_seed: int = 0,
) -> dict:
    """Assemble the Part VII method-comparison: STRIDE vs baselines on planted nulls.

    STRIDE's over-resolution rate is measured with the **reused** V5 metric
    (``empirical_over_resolution_rate`` via the adapter bridge); the baselines' rates
    come from this module. The paired McNemar and paired-bootstrap comparisons are run
    per baseline. Returns a plain-dict report (no ``mechanism`` objects) suitable for
    the ``method_comparison`` artifact.

    This function *reports* the comparison; it encodes no expectation about the sign of
    the difference. Whether STRIDE beats a baseline is an [EMPIRICAL RESULT] recorded in
    the artifact, not a hard-coded invariant.
    """
    # lazy imports: keep this module importable without mechanism on the path
    from .adapters import gate_via_production

    # STRIDE per-seed over-resolution indicators on the NULL ensemble
    stride_ind = []
    for s in seeds:
        mechs = gate_via_production(make_null_frames(s), hierarchy_config,
                                    rho_star=rho_star, protein=protein)
        driver = [m for m in mechs if driver_region_substr in m["region_id"]]
        # over-resolution = emitting a scale strictly finer than the true scale
        over = False
        if driver:
            finest = min(m["scale_index"] for m in driver)
            order = {"residue": 0, "domain": 1, "chain": 2, "protein": 3,
                     "complex": 4}
            true_idx = order.get(true_scale_level, 0)
            over = finest < true_idx
        stride_ind.append(int(over))
    stride_ind = np.asarray(stride_ind)
    stride_rate = float(stride_ind.mean()) if stride_ind.size else float("nan")

    base = baseline_over_resolution_rates(
        make_null_frames, seeds=seeds, true_scale_level=true_scale_level, z=z,
        driver_indices=driver_indices)

    from .stats_tests import mcnemar_test, paired_bootstrap_diff
    comparisons = {}
    for name, info in base.items():
        b_ind = info["indicators"]
        # McNemar on the paired binary over-resolution outcomes
        # b = STRIDE over-resolves & baseline does not; c = baseline over & STRIDE not
        b = int(np.sum((stride_ind == 1) & (b_ind == 0)))
        c = int(np.sum((stride_ind == 0) & (b_ind == 1)))
        mc = mcnemar_test(b, c)
        pb = paired_bootstrap_diff(stride_ind.astype(float),
                                   b_ind.astype(float), seed=boot_seed)
        comparisons[name] = {
            "baseline_rate": info["rate"],
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_statistic": mc.statistic, "mcnemar_p": mc.p_value,
            "paired_bootstrap_diff": pb.diff,
            "paired_bootstrap_ci": [pb.ci_lower, pb.ci_upper],
        }
    return {
        "true_scale_level": true_scale_level,
        "rho_star": float(rho_star),
        "n_seeds": int(len(seeds)),
        "stride_over_resolution_rate": stride_rate,
        "comparisons": comparisons,
    }
