"""Empirical operating characteristics + empirical-vs-predicted check (milestone V5).

Measures the *empirical* operating characteristics of the STRIDE gate on seeded
synthetic ensembles at the **calibrated** ``rho*`` (V4), and pairs each empirical
value with the **predicted** closed-form value (V3) so the Part IV faithfulness
check (roadmap V5) can be read off directly.

Independence of the three components (roadmap V5)
-------------------------------------------------
* **Prediction (V3)** and **calibration (V4)** are *fixed references*. This module
  imports and calls them; it never modifies ``predicted.py`` or the ``rho_star.yaml``
  artifacts, and it never tunes either to improve agreement.
* Empirical and predicted results **stand independently**. Where they disagree, this
  module *computes, reports, and (where possible) explains* the difference — it does
  not reshape one to match the other. Disagreements are data about the current
  implementation, not invariants of the method.

What is measured (Part IV / Part III)
-------------------------------------
* empirical FPR — ``Pr(rho_hat >= rho* | beta = 0)`` per region (matched to how
  ``rho*`` was calibrated in V4: per-region, not max-over-regions);
* empirical power — ``Pr(rho_hat >= rho*)`` at a driver region;
* empirical coverage — coverage of the production ``beta_hat`` interval (measured on
  the estimator's own energy scale; see [CHOICE] on ``standardize`` below);
* empirical rho recovery — ``rho_hat -> rho_true`` as sampling noise shrinks (the (C)
  consistency property);
* empirical hierarchy recovery — ``ell_hat*`` vs true ``ell*`` through the production
  gate at the calibrated ``rho*`` (precision/recall of the scale-selection task);
* over-resolution rate — certifying a scale finer than truth, vs ``K`` and gap;
* I2 (upward-closed passable set) and I3 (standardization invariance) checks;
* ROC/AUC of the reproducible-vs-null decision (reporting, per the roadmap — not a
  new estimand).

Separation
----------
Like ``validation.calibrate``, this module reaches production **only** through
``validation.adapters`` (lazy imports inside the functions that need it), so
``import validation`` stays production-free and only ``adapters.py`` imports
``mechanism``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence

import numpy as np

from ._seed import make_rng, spawn_seeds
from .predicted import (
    predicted_power, predicted_fpr, predicted_coverage, rho_from_params,
    over_resolution_bound, ScalePrediction, ell_min,
)

# yaml for the provenanced metrics artifact (production dependency; reused).
import yaml


# ── result containers ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OperatingPoint:
    """One (K, T, tau2, beta2) cell pairing empirical with predicted values."""

    K: int
    T: int
    tau2: float
    beta2: float
    sigma2_bar: float
    rho_star: float
    rho_true: float
    n_draws: int
    empirical_power: float
    predicted_power: float
    empirical_fpr: float
    predicted_fpr: float

    @property
    def power_diff(self) -> float:
        """Signed empirical - predicted power (reported, never constrained)."""
        return self.empirical_power - self.predicted_power

    def to_dict(self) -> dict:
        d = asdict(self)
        d["power_diff"] = self.power_diff
        return d


@dataclass(frozen=True)
class MetricsReport:
    """A provenanced empirical-vs-predicted report for one system."""

    system: str
    alpha: float
    seed: int
    points: tuple           # tuple[OperatingPoint, ...]
    ell_min_grid: tuple     # tuple[dict, ...]  (K, T, ell_min_predicted)

    def to_dict(self) -> dict:
        return {
            "system": self.system, "alpha": self.alpha, "seed": self.seed,
            "points": [p.to_dict() for p in self.points],
            "ell_min_grid": [dict(g) for g in self.ell_min_grid],
        }


# ── ensemble gate-crossing rates (empirical FPR / power) ─────────────────────
def _region_rho(out, scale_level: str, label: Optional[str] = None):
    """Extract rho_hat for a region (or all regions at a scale) from an M4 table."""
    sub = out[out["scale_level"] == scale_level]
    if label is not None:
        sub = sub[sub["label"] == label]
    return sub["rho"].to_numpy(dtype=float)


def empirical_crossing_rate(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    rho_star: float, scale_level: str, label: Optional[str] = None,
    protein: str = "",
) -> float:
    """Fraction of seeded draws whose region ``rho_hat >= rho_star``.

    With ``label=None`` this pools all regions at ``scale_level`` (the per-region
    convention used to calibrate ``rho*`` in V4) — use it for **FPR** on null
    systems. With a ``label`` it targets a single driver region — use it for
    **power**. ``make_frames(seed) -> list[frame]`` builds one system's per-run
    frames. rho_hat comes from the production estimator via the adapter bridge.
    """
    from .adapters import aggregate_via_production  # lazy: keep import mechanism-free
    hits = 0
    total = 0
    for s in seeds:
        out = aggregate_via_production(make_frames(s), hierarchy_config,
                                       protein=protein)
        rhos = _region_rho(out, scale_level, label)
        rhos = rhos[np.isfinite(rhos)]
        total += rhos.size
        hits += int(np.sum(rhos >= rho_star))
    return float(hits / total) if total else float("nan")


# ── consistency: rho_hat -> rho_true as sampling noise shrinks (C) ───────────
def empirical_rho_recovery(
    make_frames_for_sigma: Callable, hierarchy_config, *, sigma2_grid: Sequence[float],
    beta2: float, tau2: float, seeds_per_point: int, seed: int,
    scale_level: str, label: str, protein: str = "",
) -> list:
    """Mean ``rho_hat`` vs ``rho_true`` across a shrinking-sigma^2 grid (property (C)).

    ``make_frames_for_sigma(sigma2, seed) -> list[frame]`` builds a driver system with
    the given within-variance. Returns a list of dicts with ``sigma2``,
    ``mean_rho_hat``, ``rho_true``. As ``sigma2 -> 0`` the mean ``rho_hat`` should
    approach ``rho_true = beta2 / (beta2 + tau2 + sigma2)`` (consistency).
    """
    from .adapters import aggregate_via_production  # lazy
    rows = []
    for sig2 in sigma2_grid:
        seeds = spawn_seeds(seed + int(sig2 * 1e6) + 1, seeds_per_point)
        rhos = []
        for s in seeds:
            out = aggregate_via_production(make_frames_for_sigma(sig2, s),
                                           hierarchy_config, protein=protein)
            r = _region_rho(out, scale_level, label)
            if r.size:
                rhos.append(float(r[0]))
        rows.append({
            "sigma2": float(sig2),
            "mean_rho_hat": float(np.mean(rhos)) if rhos else float("nan"),
            "rho_true": float(rho_from_params(beta2, tau2, sig2)),
        })
    return rows


# ── coverage of the beta_hat interval ────────────────────────────────────────
def empirical_coverage(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    scale_level: str, label: str, z: float = 1.96, protein: str = "",
    standardize: bool = False,
) -> dict:
    """Empirical coverage of the production ``beta_hat`` interval at a driver region.

    [CHOICE] The production ``beta_hat`` lives on the energy (``A_en``) scale, and the
    M4 aggregator standardizes ``theta`` across the pooled field by default (I3), which
    rescales that beta. To measure a well-defined coverage we call the estimator with
    ``standardize=False`` so ``beta_hat`` stays in native energy units, and we score
    coverage of the **ensemble-mean** ``beta_hat`` (the quantity the per-run interval
    is an estimate of). Returns ``{mean_beta, target_beta, coverage, n}``. This
    measures interval calibration on the estimator's own scale; it does not assert a
    mapping onto the planted theta-scale beta (which folding makes non-identity).
    """
    from .adapters import aggregate_via_production  # lazy
    betas, cis = [], []
    for s in seeds:
        out = aggregate_via_production(make_frames(s), hierarchy_config,
                                       protein=protein, standardize=standardize)
        sub = out[(out["scale_level"] == scale_level) & (out["label"] == label)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        b = float(row["beta"])
        se = float(row["beta_se"])
        if not (np.isfinite(b) and np.isfinite(se)):
            continue
        betas.append(b)
        cis.append((b - z * se, b + z * se))
    if not betas:
        return {"mean_beta": float("nan"), "target_beta": float("nan"),
                "coverage": float("nan"), "n": 0}
    target = float(np.mean(betas))
    cov = float(np.mean([lo <= target <= hi for lo, hi in cis]))
    return {"mean_beta": target, "target_beta": target, "coverage": cov,
            "n": len(betas)}


# ── hierarchy recovery via the production gate ───────────────────────────────
def empirical_hierarchy_recovery(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    rho_star: float, true_scale_level: str, driver_region_substr: str,
    protein: str = "", coherence_threshold: float = 0.6,
) -> dict:
    """Precision/recall of scale selection at the calibrated ``rho*``.

    Runs the production gate (``run_aggregation`` at ``rho_star``) and, for each draw,
    checks whether a mechanism is emitted for the driver region and whether its gated
    scale equals ``true_scale_level``. Returns counts and the derived rates:
    ``emitted`` (recall of *some* claim), ``correct_scale`` (fraction whose gated
    scale matches truth), ``finer_than_truth`` and ``coarser_than_truth`` (the two
    mis-resolution directions). Uses the production ``GateConfig`` unchanged.
    """
    from .adapters import gate_via_production  # lazy
    n = len(seeds)
    emitted = correct = finer = coarser = 0
    for s in seeds:
        mechs = gate_via_production(make_frames(s), hierarchy_config,
                                    rho_star=rho_star, protein=protein,
                                    coherence_threshold=coherence_threshold)
        driver = [m for m in mechs if driver_region_substr in m["region_id"]]
        if not driver:
            continue
        emitted += 1
        # the finest emitted scale for the driver (smallest scale_index)
        idxs = [m["scale_index"] for m in driver]
        levels = {m["scale_index"]: m["scale_level"] for m in driver}
        finest = min(idxs)
        # map true_scale_level to its index via any emitted level, else compare names
        if levels[finest] == true_scale_level:
            correct += 1
        else:
            # decide finer/coarser by scale_index (0 = residue = finest)
            true_idx = _true_scale_index(driver, true_scale_level)
            if true_idx is None:
                pass
            elif finest < true_idx:
                finer += 1
            else:
                coarser += 1
    return {
        "n": n, "emitted": emitted, "correct_scale": correct,
        "finer_than_truth": finer, "coarser_than_truth": coarser,
        "recall": float(emitted / n) if n else float("nan"),
        "scale_accuracy": float(correct / emitted) if emitted else float("nan"),
    }


def _true_scale_index(driver_mechs, true_scale_level):
    for m in driver_mechs:
        if m["scale_level"] == true_scale_level:
            return m["scale_index"]
    # not among emitted; fall back to a standard 5-level ordering
    order = {"residue": 0, "domain": 1, "chain": 2, "protein": 3, "complex": 4}
    return order.get(true_scale_level)


def empirical_over_resolution_rate(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    rho_star: float, true_scale_level: str, driver_region_substr: str,
    protein: str = "",
) -> float:
    """Empirical rate of certifying a scale strictly finer than truth (§Part III)."""
    rec = empirical_hierarchy_recovery(
        make_frames, hierarchy_config, seeds=seeds, rho_star=rho_star,
        true_scale_level=true_scale_level, driver_region_substr=driver_region_substr,
        protein=protein)
    n = rec["n"]
    return float(rec["finer_than_truth"] / n) if n else float("nan")


# ── I2 / I3 invariance checks on simulated systems ───────────────────────────
def check_I2_upward_closed(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    rho_star: float, protein: str = "",
) -> float:
    """Fraction of draws whose *passable set* (rho_hat >= rho*) is upward-closed.

    (I2) says: if a region at scale ell passes, some coarsening also passes. Since the
    hierarchy is nested and region ids are path-prefixes, we check that for every
    passing region there exists a passing ancestor at every coarser scale on its path
    (the whole-system root is the maximal fallback). Returns the fraction of draws for
    which this holds (should be ~1).
    """
    from .adapters import aggregate_via_production  # lazy
    ok = 0
    for s in seeds:
        out = aggregate_via_production(make_frames(s), hierarchy_config,
                                       protein=protein)
        passing = out[out["rho"] >= rho_star]
        if passing.empty:
            ok += 1  # vacuously upward-closed
            continue
        ids = set(out["region_id"])
        rho_by_id = dict(zip(out["region_id"], out["rho"]))
        good = True
        for rid in passing["region_id"]:
            # every ancestor prefix present in the table must also pass at the root;
            # concretely: the root region (no "/") must pass (maximal fallback).
            root = rid.split("/")[0]
            # find the coarsest region on this path (the shortest prefix present)
            ancestors = [q for q in ids if rid == q or rid.startswith(q + "/")]
            coarsest = min(ancestors, key=lambda q: q.count("/"))
            if rho_by_id[coarsest] < rho_star:
                good = False
                break
        ok += int(good)
    return float(ok / len(seeds)) if seeds else float("nan")


def check_I3_standardization_invariance(
    make_frames: Callable, hierarchy_config, *, seeds: Sequence[int],
    scale_factor: float = 7.0, protein: str = "",
) -> float:
    """Max ``|rho_hat(theta) - rho_hat(c*theta)|`` over regions/draws (should be ~0).

    (I3): ``ell_hat*`` depends on data only through the *standardized* effect field, so
    rescaling the effect measurement by a constant ``c`` must leave ``rho_hat``
    unchanged (the aggregator standardizes before ``A_en``). The effect *and its
    sampling uncertainty* share units, so a faithful rescaling multiplies both the
    effect column and the standard-error columns by ``c`` (rescaling the value alone
    would change the signal/noise ratio and is not what standardization means).
    Returns the maximum absolute rho difference across all regions and draws — a
    near-zero value confirms invariance.
    """
    from .adapters import aggregate_via_production  # lazy
    c = float(scale_factor)
    worst = 0.0
    for s in seeds:
        frames = make_frames(s)
        scaled = []
        for f in frames:
            g = f.copy()
            g["r"] = g["r"] * c
            if "abs_r" in g.columns:
                g["abs_r"] = np.abs(g["r"])
            for se_col in ("theta_se", "theta_bootstrap_se"):
                if se_col in g.columns:
                    g[se_col] = g[se_col] * c
            scaled.append(g)
        a = aggregate_via_production(frames, hierarchy_config, protein=protein)
        b = aggregate_via_production(scaled, hierarchy_config, protein=protein)
        m = a.merge(b, on="region_id", suffixes=("_a", "_b"))
        diff = np.abs(m["rho_a"].to_numpy() - m["rho_b"].to_numpy())
        diff = diff[np.isfinite(diff)]
        if diff.size:
            worst = max(worst, float(diff.max()))
    return worst


# ── ROC / AUC of the reproducible-vs-null decision (reporting) ───────────────
def roc_auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """AUC via the Mann-Whitney U statistic (probability a positive outranks a null).

    [ROADMAP: reporting summary of the scale-selection task, not a new estimand.]
    ``scores_pos`` are ``rho_hat`` at true driver regions, ``scores_neg`` at null
    regions. Returns AUC in ``[0, 1]``; 0.5 = chance, 1.0 = perfect separation.
    """
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # rank-based U (handles ties at 0.5)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _assign_tie_ranks(allv, ranks)
    r_pos = ranks[:pos.size].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    """In-place average-rank correction for ties (stable)."""
    order = np.argsort(values, kind="mergesort")
    sorted_v = values[order]
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = np.mean([ranks[order[k]] for k in range(i, j + 1)])
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


# ── the central deliverable: empirical-vs-predicted table ────────────────────
def operating_point(
    make_null_frames: Callable, make_driver_frames: Callable, hierarchy_config, *,
    K: int, T: int, tau2: float, beta2: float, sigma2_bar: float, rho_star: float,
    scale_level: str, driver_label: str, seeds_null: Sequence[int],
    seeds_driver: Sequence[int], alpha: float = 0.05, protein: str = "",
) -> OperatingPoint:
    """Compute one empirical-vs-predicted operating point.

    Empirical FPR/power come from the production estimator at ``rho_star`` (fixed V4
    reference); predicted FPR/power come from the V3 closed form (fixed reference).
    The two are *reported side by side*; neither is adjusted toward the other.
    """
    emp_fpr = empirical_crossing_rate(
        make_null_frames, hierarchy_config, seeds=seeds_null, rho_star=rho_star,
        scale_level=scale_level, label=None, protein=protein)
    emp_pow = empirical_crossing_rate(
        make_driver_frames, hierarchy_config, seeds=seeds_driver, rho_star=rho_star,
        scale_level=scale_level, label=driver_label, protein=protein)
    pred_pow = predicted_power(beta2, tau2, sigma2_bar, K=K, rho_star=rho_star)
    return OperatingPoint(
        K=K, T=T, tau2=tau2, beta2=beta2, sigma2_bar=sigma2_bar,
        rho_star=rho_star, rho_true=rho_from_params(beta2, tau2, sigma2_bar),
        n_draws=len(seeds_driver),
        empirical_power=emp_pow, predicted_power=pred_pow,
        empirical_fpr=emp_fpr, predicted_fpr=predicted_fpr(alpha))


def ell_min_grid(
    scales_by_KT: Callable, rho_star: float, Ks: Sequence[int], Ts: Sequence[int],
) -> list:
    """Predicted ``ell_min`` over a (K, T) grid (V3 reference; no empirical input).

    ``scales_by_KT(K, T) -> list[ScalePrediction]`` supplies the per-scale predicted
    ``rho`` at that (K, T). Returns a list of ``{K, T, ell_min}`` using the V3
    ``ell_min`` selector. This is the predicted grid the roadmap lists as an output;
    empirical hierarchy recovery is reported separately.
    """
    grid = []
    for K in Ks:
        for T in Ts:
            lm = ell_min(scales_by_KT(K, T), rho_star)
            grid.append({"K": int(K), "T": int(T),
                         "ell_min": (None if lm is None else int(lm))})
    return grid


# ── provenanced artifact I/O ─────────────────────────────────────────────────
def write_metrics_report(report: MetricsReport, path: str) -> None:
    """Write a provenanced metrics report (empirical-vs-predicted table + ell_min)."""
    payload = {
        "provenance": {"system": report.system, "alpha": report.alpha,
                       "seed": report.seed},
        "operating_points": [p.to_dict() for p in report.points],
        "ell_min_grid": [dict(g) for g in report.ell_min_grid],
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)


def load_metrics_report(path: str) -> dict:
    """Load a metrics report artifact as a plain dict (for inspection / tests)."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
