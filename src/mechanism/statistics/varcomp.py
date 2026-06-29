"""Variance-components estimation for the reproducibility model (milestone M4).

Implements §2.2 of MATHEMATICAL_SPECIFICATION.md: the one-way random-effects
("generalizability-theory") model for a regional statistic across replicates,

    Theta_R^(k) = beta_R + gamma_R^(k) + e_R^(k),
        gamma_R^(k) ~ (0, tau_R^2),   e_R^(k) ~ (0, sigma_R^2),

estimating ``(beta_R, tau_R^2)`` with ``sigma_R^2`` (the within-replicate
sampling variance) **propagated as a known quantity from §2.1** (effective-N /
block bootstrap). Because the within-replicate variances are *known* and not
estimated from between-replicate replication, this is exactly the Normal-Normal
random-effects **meta-analysis** model with known within-study variances:

    y_k := Theta_R^(k) ~ N(beta, tau^2 + v_k),   v_k := sigma_{R,k}^2 known.

Estimators provided (selected per the spec's K-dependent rule, §2.2):

* **Paule-Mandel** (``method="paule_mandel"``) — the recommended moment
  estimator of ``tau^2`` for ``K >= 5`` (REML-consistent, no Normal-likelihood
  iteration; standard in meta-analysis).
* **Bayesian half-Normal prior on tau** (``method="bayesian"``) — the spec's
  small-``K`` path (the ``K = 3`` regime). The posterior is computed by **1-D
  numerical quadrature over tau** with ``beta`` marginalised analytically (it is
  Gaussian given ``tau``). No MCMC, no extra dependency, fully deterministic.
* **DerSimonian-Laird** (``method="dersimonian_laird"``) — closed-form moment
  estimator; always available as a fast fallback / initialiser and at the
  ``tau^2 = 0`` boundary.

This module computes variance components only. It does **not** compute the
reproducibility coefficient (see ``reproducibility.py``), the resolution gate or
profile (M5), or the calibrated threshold ``rho*`` (validation phase). Nothing in
M0-M3 is modified.

References
----------
DerSimonian & Laird (1986); Paule & Mandel (1982); Veroniki et al. (2016) on
between-study variance estimators; standard Bayesian random-effects
meta-analysis with a half-Normal prior on the heterogeneity SD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# numpy 2.0 renamed ``trapz`` -> ``trapezoid``; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Numerical safeguards -------------------------------------------------------
_VAR_FLOOR = 1e-12        # smallest admissible within-replicate variance v_k
_TAU2_FLOOR = 0.0         # tau^2 is a variance: non-negative
_GRID_POINTS = 401        # quadrature nodes for the Bayesian tau posterior
_REML_K_THRESHOLD = 5     # spec §2.2: REML for K>=5, Bayesian for small K


@dataclass(frozen=True)
class VarCompResult:
    """Estimated components of the one-way random-effects model.

    Attributes
    ----------
    beta : float
        Pooled (reproducible population) effect ``hat beta_R``.
    beta_se : float
        Standard error of ``beta`` (absorbs ``tau^2`` and the known ``v_k``).
    tau2 : float
        Between-replicate variance ``hat tau_R^2`` (non-stationarity).
    tau2_sd : float
        Posterior SD of ``tau^2`` (Bayesian path) or ``nan`` (moment paths).
    sigma2_bar : float
        Mean known within-replicate sampling variance ``(1/K) sum_k v_k``.
    n_replicates : int
        Number of replicates ``K`` used.
    method : str
        Estimator actually used: ``paule_mandel``, ``bayesian``,
        ``dersimonian_laird``, or ``undefined``.
    status : str
        ``ok``; ``tau2_boundary`` (tau^2 hit the zero boundary);
        ``gate_uncertain`` (K < 5: tau^2 weakly identified, Bayesian prior
        load-bearing); ``k_too_small`` (K < 2: tau^2 unidentifiable).
    """

    beta: float
    beta_se: float
    tau2: float
    tau2_sd: float
    sigma2_bar: float
    n_replicates: int
    method: str
    status: str


# ── building blocks ──────────────────────────────────────────────────────────
def _sanitize(y: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if y.shape != v.shape:
        raise ValueError("y and v must have the same shape")
    # drop non-finite replicates (e.g. degenerate within-run variance)
    mask = np.isfinite(y) & np.isfinite(v)
    y, v = y[mask], v[mask]
    v = np.maximum(v, _VAR_FLOOR)
    return y, v


def _pooled(y: np.ndarray, v: np.ndarray, tau2: float) -> tuple[float, float]:
    """Inverse-variance pooled mean and its variance at a given ``tau2``."""
    w = 1.0 / (v + tau2)
    sw = w.sum()
    beta = float((w * y).sum() / sw)
    beta_var = float(1.0 / sw)
    return beta, beta_var


def dersimonian_laird(y: np.ndarray, v: np.ndarray) -> float:
    """Closed-form DerSimonian-Laird estimate of ``tau^2`` (>= 0)."""
    y, v = _sanitize(y, v)
    k = y.size
    if k < 2:
        return float("nan")
    w = 1.0 / v
    sw = w.sum()
    beta_fe = (w * y).sum() / sw
    q = float((w * (y - beta_fe) ** 2).sum())
    c = float(sw - (w ** 2).sum() / sw)
    if c <= 0:
        return 0.0
    return max((q - (k - 1)) / c, _TAU2_FLOOR)


def paule_mandel(y: np.ndarray, v: np.ndarray, max_iter: int = 200,
                 tol: float = 1e-10) -> float:
    """Iterative Paule-Mandel estimate of ``tau^2`` (>= 0).

    Solves ``sum_k (y_k - beta(tau^2))^2 / (v_k + tau^2) = K - 1`` for ``tau^2``,
    where ``beta(tau^2)`` is the inverse-variance pooled mean. The generalised-Q
    statistic is monotone decreasing in ``tau^2``, so a simple bracketed bisection
    is robust and derivative-free.
    """
    y, v = _sanitize(y, v)
    k = y.size
    if k < 2:
        return float("nan")

    def gen_q(tau2: float) -> float:
        beta, _ = _pooled(y, v, tau2)
        w = 1.0 / (v + tau2)
        return float((w * (y - beta) ** 2).sum())

    target = k - 1
    # If even tau^2 = 0 gives Q <= K-1, there is no positive root: tau^2 = 0.
    if gen_q(0.0) <= target:
        return 0.0
    lo, hi = 0.0, max(np.var(y), 1e-6)
    # expand the upper bracket until Q(hi) <= target
    for _ in range(100):
        if gen_q(hi) <= target:
            break
        hi *= 2.0
    else:
        return hi  # extremely heterogeneous; return the (large) bracket end
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        q = gen_q(mid)
        if abs(q - target) < tol:
            return max(mid, _TAU2_FLOOR)
        if q > target:
            lo = mid
        else:
            hi = mid
    return max(0.5 * (lo + hi), _TAU2_FLOOR)


def _default_prior_scale(y: np.ndarray, v: np.ndarray) -> float:
    """Weakly-informative half-Normal scale for the heterogeneity SD ``tau``.

    Set to the order of the observed dispersion: the larger of the ordinary SD of
    the replicate statistics and the typical within-replicate SD. Documented and
    overridable (``prior_scale``); the spec fixes the prior *family* (half-Normal
    on ``tau``), not its scale.

    The dispersion is **deliberately non-robust** (ordinary SD, not MAD). Under
    the I1 scenario a single replicate carries the signal and the rest are ~0; a
    robust scale would treat that spike as an outlier, collapse the prior scale,
    and spuriously shrink ``tau^2`` toward zero — certifying residue-level
    reproducibility that is not there (anti-conservative). The ordinary SD admits
    the large between-replicate variance the spike implies, keeping the gate
    conservative, while still collapsing to ~0 when replicates agree.
    """
    if y.size >= 2:
        sd = float(np.std(y, ddof=1))
    else:
        sd = 0.0
    within = float(np.sqrt(np.mean(v)))
    scale = max(sd, within, 1e-6)
    return float(scale)


def bayesian_halfnormal(
    y: np.ndarray, v: np.ndarray, *, prior_scale: Optional[float] = None,
    grid_points: int = _GRID_POINTS,
) -> tuple[float, float, float, float]:
    """Bayesian random-effects fit with a half-Normal prior on ``tau``.

    Marginalises ``beta`` analytically (flat prior; Gaussian given ``tau``) and
    integrates the posterior over ``tau`` by trapezoidal quadrature on a grid.
    Deterministic (no MCMC).

    Returns
    -------
    (beta_mean, beta_sd, tau2_mean, tau2_sd)
        Posterior means and SDs of ``beta`` and ``tau^2``.
    """
    y, v = _sanitize(y, v)
    k = y.size
    if k < 2:
        return (float(y[0]) if k == 1 else float("nan"),
                float(np.sqrt(v[0])) if k == 1 else float("nan"),
                float("nan"), float("nan"))

    s = _default_prior_scale(y, v) if prior_scale is None else float(prior_scale)
    tau_max = max(5.0 * s, 5.0 * float(np.sqrt(v.max())), 5.0 * float(np.std(y)))
    tau_max = max(tau_max, 1e-3)
    taus = np.linspace(0.0, tau_max, grid_points)
    tau2s = taus ** 2

    # log marginal likelihood at each tau (beta integrated out, flat prior)
    log_ml = np.empty_like(taus)
    beta_tau = np.empty_like(taus)
    betavar_tau = np.empty_like(taus)
    for j, t2 in enumerate(tau2s):
        veff = v + t2
        w = 1.0 / veff
        sw = w.sum()
        beta = (w * y).sum() / sw
        resid = float((w * (y - beta) ** 2).sum())
        # -0.5*sum log(2*pi*veff)  - 0.5*resid  + 0.5*log(2*pi/sw)
        log_ml[j] = (-0.5 * np.sum(np.log(2.0 * np.pi * veff))
                     - 0.5 * resid + 0.5 * np.log(2.0 * np.pi / sw))
        beta_tau[j] = beta
        betavar_tau[j] = 1.0 / sw

    # half-Normal(scale=s) log prior on tau (>=0): -tau^2/(2 s^2)  (+const)
    log_prior = -tau2s / (2.0 * s * s)
    log_post = log_ml + log_prior
    log_post -= log_post.max()
    post = np.exp(log_post)

    # normalise over the tau grid
    norm = _trapz(post, taus)
    if not np.isfinite(norm) or norm <= 0:
        # fall back to a moment estimate if the grid is degenerate
        t2 = paule_mandel(y, v)
        beta, bvar = _pooled(y, v, t2)
        return beta, float(np.sqrt(bvar)), float(t2), float("nan")
    p = post / norm

    tau2_mean = float(_trapz(p * tau2s, taus))
    tau2_e2 = float(_trapz(p * tau2s ** 2, taus))
    tau2_sd = float(np.sqrt(max(tau2_e2 - tau2_mean ** 2, 0.0)))

    beta_mean = float(_trapz(p * beta_tau, taus))
    beta_e2 = float(_trapz(p * (betavar_tau + beta_tau ** 2), taus))
    beta_sd = float(np.sqrt(max(beta_e2 - beta_mean ** 2, 0.0)))

    return beta_mean, beta_sd, tau2_mean, tau2_sd


# ── top-level dispatcher ─────────────────────────────────────────────────────
def fit(
    y, v, *, method: str = "auto", prior_scale: Optional[float] = None,
    reml_k_threshold: int = _REML_K_THRESHOLD,
) -> VarCompResult:
    """Fit the random-effects model and return ``VarCompResult``.

    Parameters
    ----------
    y : array_like
        Regional statistics across replicates, ``Theta_R^(k)``.
    v : array_like
        **Known** within-replicate sampling variances ``sigma_{R,k}^2`` (§2.1).
    method : {"auto", "paule_mandel", "bayesian", "dersimonian_laird"}
        ``"auto"`` (spec §2.2): Paule-Mandel for ``K >= reml_k_threshold``,
        Bayesian half-Normal for ``2 <= K < threshold``, undefined for ``K < 2``.
    prior_scale : float, optional
        Half-Normal scale for ``tau`` (Bayesian path); default data-driven.
    reml_k_threshold : int
        ``K`` at/above which the moment (REML-like) path is used.
    """
    y, v = _sanitize(y, v)
    k = int(y.size)
    sigma2_bar = float(np.mean(v)) if k else float("nan")

    if k < 2:
        beta = float(y[0]) if k == 1 else float("nan")
        beta_se = float(np.sqrt(v[0])) if k == 1 else float("nan")
        return VarCompResult(beta, beta_se, float("nan"), float("nan"),
                             sigma2_bar, k, "undefined", "k_too_small")

    chosen = method
    if method == "auto":
        chosen = "paule_mandel" if k >= reml_k_threshold else "bayesian"

    if chosen == "bayesian":
        beta, beta_se, tau2, tau2_sd = bayesian_halfnormal(
            y, v, prior_scale=prior_scale)
        status = "gate_uncertain" if k < reml_k_threshold else "ok"
    elif chosen == "dersimonian_laird":
        tau2 = dersimonian_laird(y, v)
        beta, beta_var = _pooled(y, v, tau2)
        beta_se, tau2_sd = float(np.sqrt(beta_var)), float("nan")
        status = "tau2_boundary" if tau2 <= _VAR_FLOOR else "ok"
    elif chosen == "paule_mandel":
        tau2 = paule_mandel(y, v)
        beta, beta_var = _pooled(y, v, tau2)
        beta_se, tau2_sd = float(np.sqrt(beta_var)), float("nan")
        status = "tau2_boundary" if tau2 <= _VAR_FLOOR else "ok"
    else:
        raise ValueError(f"unknown method {method!r}")

    if chosen != "bayesian" and tau2 <= _VAR_FLOOR and k < reml_k_threshold:
        status = "gate_uncertain"
    return VarCompResult(float(beta), float(beta_se), float(tau2),
                         float(tau2_sd), sigma2_bar, k, chosen, status)
