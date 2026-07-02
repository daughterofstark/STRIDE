"""Empirical rho* calibration via null surrogates (milestone V4).

Implements the specification's calibration (Part V step 11): ``rho* = upper-alpha
quantile of rho_hat under B null surrogates``, giving the (Cal) FPR-control
guarantee ``Pr(rho_hat >= rho* | beta_R = 0) = alpha`` by construction.

Canonical procedure (surrogate-based) — [SPEC] / [ROADMAP]
---------------------------------------------------------
1. Draw an **ensemble** of null (``beta = 0``) base datasets for a system, each a set
   of ``K`` per-run effect frames (from the V1 Tier-A or V2 Tier-B generators —
   reused, never re-implemented).
2. For each base, draw ``surr_per_base`` **null surrogates** (replicate-label
   permutation, the canonical field-level scheme; phase randomization is the
   series-level scheme in :mod:`validation.surrogates`), each destroying
   cross-replicate reproducibility while preserving the sampling-noise budget.
3. For each surrogate, compute ``rho_hat`` per region/scale via the **production**
   estimator (``aggregate_via_production`` → M4 ``aggregate_reproducibility``), and
   pool across all bases and surrogates.
4. ``rho*(scale) = upper-alpha quantile`` of the pooled surrogate ``rho_hat`` at that
   scale. This is the calibrated threshold — it is **not** hard-coded; it is entirely
   determined by the surrogate null quantile.

Why an *ensemble* of bases (not a single dataset)
-------------------------------------------------
[KNOWN LIMITATION, characterized in V4] Surrogates of a *single* base dataset
capture only that dataset's within-permutation variability, which is narrower than
the true null spread across independent ``beta = 0`` realizations; a ``rho*`` from a
single base under-covers (out-of-sample FPR > alpha). Pooling surrogates across an
ensemble of null bases captures the between-realization variability and restores the
(Cal) guarantee ``FPR <= alpha`` (validated in the V4 tests). This is an observed
property of the folded-energy estimator (``A_en`` depends only on magnitudes, which
replicate-label permutation preserves), documented honestly without any change to the
specification or production.

Independent validation (does NOT replace the surrogate calibration) — [CHOICE]
------------------------------------------------------------------------------
Direct ``beta = 0`` generator draws provide an *independent* null against which the
surrogate-calibrated ``rho*`` is checked (out-of-sample FPR ≤ alpha on disjoint
draws). Per the maintainer's V4 instruction, generator nulls **validate** the
surrogate calibration; they do not produce the calibrated artifact. The two roles
are kept explicit: :func:`calibrate_rho_star` (surrogate; canonical, produces the
artifact) vs :func:`generator_null_rho` (generator; validation only).

Determinism
-----------
All randomness flows through :mod:`validation._seed`. Calibration and evaluation use
**disjoint** seed streams (``spawn_seeds``), enforced by the tests, so a rho* is
reproducible from its recorded ``(system, K, T, alpha, B, seed)`` provenance.

Separation
----------
This module reaches production **only** through :func:`validation.adapters.
aggregate_via_production`; it never imports ``mechanism`` directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

from ._seed import make_rng, spawn_seeds
from .surrogates import permute_replicate_labels
# NOTE: the production bridge (validation.adapters.aggregate_via_production) is
# imported LAZILY inside the functions that need it, so that ``import
# validation.calibrate`` (and ``import validation``) does not pull ``mechanism`` onto
# the path. This preserves the V0/V1 guarantee that the validation package imports
# production-free; only the calibration *call* touches production. [REPO invariant]

# yaml is a production dependency (used by mechanism.config); reused here for the
# provenanced artifact, as the roadmap specifies ("delivered as data, not code").
import yaml


# ── result container ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CalibrationResult:
    """Calibrated rho* for one system, with provenance and uncertainty.

    Attributes
    ----------
    system : str
        System name (the generator spec ``name``).
    K, T : int
        Replicate count and per-replicate length (``T`` may be 0 for field-level
        Tier-A calibration, which plants sigma^2 directly).
    alpha : float
        Target false-resolution rate.
    B : int
        Number of null surrogates drawn.
    seed : int
        Master calibration seed (the surrogate stream derives from it).
    surrogate : str
        Which surrogate scheme produced the null ("permute_labels" | "phase_random").
    rho_star : dict
        ``scale_level -> calibrated rho*`` (the upper-alpha quantile per scale).
    rho_star_ci : dict
        ``scale_level -> (lo, hi)`` bootstrap CI on the quantile (uncertainty).
    n_null : dict
        ``scale_level -> number of surrogate rho_hat values pooled`` at that scale.
    """

    system: str
    K: int
    T: int
    alpha: float
    B: int
    seed: int
    surrogate: str
    rho_star: dict
    rho_star_ci: dict
    n_null: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        return cls(
            system=d["system"], K=int(d["K"]), T=int(d["T"]),
            alpha=float(d["alpha"]), B=int(d["B"]), seed=int(d["seed"]),
            surrogate=d["surrogate"],
            rho_star={k: float(v) for k, v in d["rho_star"].items()},
            rho_star_ci={k: (float(v[0]), float(v[1]))
                         for k, v in d["rho_star_ci"].items()},
            n_null={k: int(v) for k, v in d["n_null"].items()},
        )


# ── quantile with bootstrap uncertainty ──────────────────────────────────────
def upper_alpha_quantile(values: np.ndarray, alpha: float) -> float:
    """Upper-``alpha`` quantile = the ``(1 - alpha)`` quantile [SPEC step 11]."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.quantile(v, 1.0 - alpha))


def _quantile_bootstrap_ci(values: np.ndarray, alpha: float, *,
                           n_boot: int = 500, ci: float = 0.95,
                           rng: Optional[np.random.Generator] = None) -> tuple:
    """Bootstrap CI on the upper-alpha quantile (calibration uncertainty)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        q = upper_alpha_quantile(v, alpha)
        return (q, q)
    if rng is None:
        rng = make_rng(0)
    qs = np.empty(n_boot)
    n = v.size
    for b in range(n_boot):
        sample = v[rng.integers(0, n, size=n)]
        qs[b] = np.quantile(sample, 1.0 - alpha)
    lo = float(np.quantile(qs, (1.0 - ci) / 2.0))
    hi = float(np.quantile(qs, 1.0 - (1.0 - ci) / 2.0))
    return (lo, hi)


# ── the canonical surrogate calibration ──────────────────────────────────────
def surrogate_null_rho(
    base_frames: Sequence, hierarchy_config, *, B: int, seed: int,
    protein: str = "", scales: Optional[Sequence[str]] = None,
) -> dict:
    """Pooled surrogate-null ``rho_hat`` per scale from **one** base dataset.

    Draws ``B`` replicate-label-permutation surrogates of ``base_frames`` and, for
    each, computes ``rho_hat`` at every region/scale via the production estimator,
    pooling the values by ``scale_level``. Returns ``{scale_level: np.ndarray}``.

    Deterministic in ``seed``: the ``B`` surrogate RNGs derive from ``spawn_seeds``.

    [KNOWN LIMITATION] Surrogates of a *single* base dataset capture only the
    within-dataset permutation variability of ``rho_hat``; they do **not** capture
    the between-realization sampling variability across independent ``beta = 0``
    datasets. A ``rho*`` calibrated from a single base therefore under-estimates the
    true null spread and yields out-of-sample FPR **above** ``alpha``. The canonical
    calibration (:func:`calibrate_rho_star`) pools surrogates across an *ensemble* of
    null base datasets, which restores FPR <= alpha (validated in the V4 tests). This
    function is the per-dataset building block, not the calibration entry point.
    """
    from .adapters import aggregate_via_production  # lazy: keep import mechanism-free
    surrogate_seeds = spawn_seeds(seed, B)
    pooled: dict = {}
    for b in range(B):
        rng = make_rng(surrogate_seeds[b])
        surr = permute_replicate_labels(base_frames, rng)
        out = aggregate_via_production(surr, hierarchy_config, protein=protein)
        for lvl, sub in out.groupby("scale_level"):
            if scales is not None and lvl not in scales:
                continue
            pooled.setdefault(lvl, []).extend(
                float(x) for x in sub["rho"].to_numpy() if np.isfinite(x))
    return {lvl: np.asarray(vals, dtype=float) for lvl, vals in pooled.items()}


def ensemble_surrogate_null_rho(
    make_base_frames, hierarchy_config, *, base_seeds: Sequence[int],
    surr_per_base: int, seed: int, protein: str = "",
    scales: Optional[Sequence[str]] = None,
) -> dict:
    """Pooled surrogate-null ``rho_hat`` per scale across an **ensemble** of bases.

    For each null base dataset (built by ``make_base_frames(base_seed)``), draw
    ``surr_per_base`` replicate-label-permutation surrogates, compute ``rho_hat`` via
    the production estimator, and pool across *all* bases and surrogates. Pooling
    across independent ``beta = 0`` bases captures the between-realization sampling
    variability that a single-base surrogate misses, so the upper-``alpha`` quantile
    of this pooled null gives proper out-of-sample FPR control (the (Cal) property).

    ``make_base_frames(base_seed) -> list[frame]`` builds one null (``beta = 0``)
    system's per-run frames. Deterministic in ``seed`` and ``base_seeds``.
    Returns ``{scale_level: np.ndarray}``.
    """
    from .adapters import aggregate_via_production  # lazy: keep import mechanism-free
    pooled: dict = {}
    for i, bs in enumerate(base_seeds):
        base_frames = make_base_frames(bs)
        surr_seeds = spawn_seeds(seed * 7919 + i, surr_per_base)
        for ss in surr_seeds:
            surr = permute_replicate_labels(base_frames, make_rng(ss))
            out = aggregate_via_production(surr, hierarchy_config, protein=protein)
            for lvl, sub in out.groupby("scale_level"):
                if scales is not None and lvl not in scales:
                    continue
                pooled.setdefault(lvl, []).extend(
                    float(x) for x in sub["rho"].to_numpy() if np.isfinite(x))
    return {lvl: np.asarray(vals, dtype=float) for lvl, vals in pooled.items()}


def calibrate_rho_star(
    make_base_frames, hierarchy_config, *, system: str, K: int, T: int,
    base_seeds: Sequence[int], surr_per_base: int = 10,
    alpha: float = 0.05, seed: int = 0, protein: str = "",
    scales: Optional[Sequence[str]] = None, ci_boot: int = 500,
) -> CalibrationResult:
    """Calibrate ``rho*`` as the upper-alpha quantile of the ensemble surrogate null.

    This is the **canonical** V4 calibration [SPEC step 11]. ``rho*`` is produced
    entirely by the surrogate-null quantile — no threshold is hard-coded. The null is
    the *ensemble* surrogate null (surrogates pooled across independent ``beta = 0``
    base datasets), which captures the between-realization variability needed for the
    (Cal) out-of-sample FPR guarantee. Calibration uncertainty is a bootstrap CI on
    the quantile.

    ``make_base_frames(base_seed) -> list[frame]`` builds one null base system's
    per-run frames (from the V1/V2 generators — reused, not re-implemented). ``B`` in
    the recorded provenance is ``len(base_seeds) * surr_per_base`` (total surrogates).

    Returns a :class:`CalibrationResult`.
    """
    null = ensemble_surrogate_null_rho(
        make_base_frames, hierarchy_config, base_seeds=base_seeds,
        surr_per_base=surr_per_base, seed=seed, protein=protein, scales=scales)
    ci_rng = make_rng(spawn_seeds(seed, 1)[0])
    rho_star: dict = {}
    rho_star_ci: dict = {}
    n_null: dict = {}
    for lvl, vals in null.items():
        rho_star[lvl] = upper_alpha_quantile(vals, alpha)
        rho_star_ci[lvl] = _quantile_bootstrap_ci(vals, alpha, n_boot=ci_boot,
                                                   rng=ci_rng)
        n_null[lvl] = int(np.asarray(vals).size)
    return CalibrationResult(
        system=system, K=int(K), T=int(T), alpha=float(alpha),
        B=int(len(base_seeds) * surr_per_base), seed=int(seed),
        surrogate="permute_labels", rho_star=rho_star,
        rho_star_ci=rho_star_ci, n_null=n_null)


# ── independent validation via generator beta=0 draws (does NOT calibrate) ────
def generator_null_rho(
    make_null_frames, hierarchy_config, *, seeds: Sequence[int], protein: str = "",
    scales: Optional[Sequence[str]] = None,
) -> dict:
    """Pooled ``rho_hat`` per scale from independent generator ``beta=0`` draws.

    ``make_null_frames(seed) -> list[frame]`` builds one null (beta=0) system's
    per-run frames for the given seed. Used to build an **independent** null against
    which a surrogate-calibrated ``rho*`` is validated (out-of-sample FPR); it is
    NOT the calibration reference. Returns ``{scale_level: np.ndarray}``.
    """
    from .adapters import aggregate_via_production  # lazy: keep import mechanism-free
    pooled: dict = {}
    for s in seeds:
        frames = make_null_frames(s)
        out = aggregate_via_production(frames, hierarchy_config, protein=protein)
        for lvl, sub in out.groupby("scale_level"):
            if scales is not None and lvl not in scales:
                continue
            pooled.setdefault(lvl, []).extend(
                float(x) for x in sub["rho"].to_numpy() if np.isfinite(x))
    return {lvl: np.asarray(vals, dtype=float) for lvl, vals in pooled.items()}


def empirical_fpr(null_rho: np.ndarray, rho_star: float) -> float:
    """Fraction of null ``rho_hat`` at or above ``rho_star`` (empirical FPR)."""
    v = np.asarray(null_rho, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v >= rho_star))


# ── provenanced artifact I/O (rho_star.yaml) ─────────────────────────────────
def write_rho_star_yaml(result: CalibrationResult, path: str) -> None:
    """Write a provenanced ``rho_star.yaml`` keyed by (system, K, T, alpha, B, seed).

    The artifact is *data, not code* (roadmap V4). CI tuples are stored as lists for
    round-trip-stable YAML.
    """
    d = result.to_dict()
    d["rho_star_ci"] = {k: [float(v[0]), float(v[1])]
                        for k, v in result.rho_star_ci.items()}
    payload = {
        "provenance": {
            "system": result.system, "K": result.K, "T": result.T,
            "alpha": result.alpha, "B": result.B, "seed": result.seed,
            "surrogate": result.surrogate,
        },
        "rho_star": {k: float(v) for k, v in result.rho_star.items()},
        "rho_star_ci": d["rho_star_ci"],
        "n_null": {k: int(v) for k, v in result.n_null.items()},
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)


def load_rho_star_yaml(path: str) -> CalibrationResult:
    """Load a ``rho_star.yaml`` artifact back into a :class:`CalibrationResult`."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    prov = payload["provenance"]
    return CalibrationResult(
        system=prov["system"], K=int(prov["K"]), T=int(prov["T"]),
        alpha=float(prov["alpha"]), B=int(prov["B"]), seed=int(prov["seed"]),
        surrogate=prov.get("surrogate", "permute_labels"),
        rho_star={k: float(v) for k, v in payload["rho_star"].items()},
        rho_star_ci={k: (float(v[0]), float(v[1]))
                     for k, v in payload["rho_star_ci"].items()},
        n_null={k: int(v) for k, v in payload["n_null"].items()},
    )
