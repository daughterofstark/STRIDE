"""Autocorrelated-process generators for Tier-B validation (milestone V2).

**Pure** module: imports **nothing** from ``mechanism``. All randomness flows
through :mod:`validation._seed`. It produces the *time series* (``V(t)`` and
``d_i(t)``) that the §2.1 sampling-noise chain consumes, so that Tier B can
validate that chain end-to-end through the **production** M1/M2 stack
(``mechanism.statistics.effective_sample_size`` / ``bootstrap_correlation`` /
``pearson_both``) rather than planting the effect field directly (that is Tier A,
V1).

What this validates (spec §2.1)
-------------------------------
* ``theta`` = Pearson ``r`` between ``V`` and ``d`` (recovered by the production
  correlation);
* ``N_eff = T / (2 tau_int)`` with ``tau_int`` the integrated autocorrelation time;
* the autocorrelation-corrected sampling variance ``sigma^2 = (1 - theta^2)^2 /
  N_eff`` and its block-bootstrap refinement (M2).

Analytic anchors
----------------
* **AR(1)** ``x_t = phi x_{t-1} + eps_t`` has integrated autocorrelation time
  ``tau_int = 1/2 * (1 + phi) / (1 - phi)`` (the exact quantity the production M1
  ``integrated_autocorr_time`` recovers on the raw series; verified in tests).
* **OU** (Ornstein–Uhlenbeck) sampled at step ``dt`` is an AR(1) with
  ``phi = exp(-theta_ou * dt)``; same ``tau_int`` formula in frames.

[KNOWN LIMITATION — characterized, not a spec claim] The production
``effective_sample_size`` estimates ``tau_int`` on the **product series**
``z(t) = (V - mean V)(d - mean d)``, whose autocorrelation differs from the raw
series'. So the clean AR(1) ``tau_int`` formula anchors the **raw** series (tested
directly), while the product-series ``tau_int`` / ``N_eff`` is a derived quantity
that Tier B *characterizes* (e.g. it shrinks with ``phi``) rather than asserting a
closed form. Separately, Sokal windowing **under-estimates** ``tau_int`` for
slow-mixing series on short trajectories; Tier B tests this direction honestly.

Misspecified processes (roadmap V2; robustness property R)
----------------------------------------------------------
Deliberate departures from the AR(1)/Gaussian assumptions so the generator cannot
secretly match the estimator's ideal regime (validation risk VR1):

* **heavy-tailed** innovations (Student-t) — same linear autocorrelation, fat tails;
* **non-AR(1)** linear structure (AR(2), including oscillatory) — a different
  autocorrelation shape;
* **slow-mixing** (``phi -> 1``) on short trajectories — where the IAT estimate is
  known to be optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ._seed import make_rng, spawn_seeds


# ── innovation families ──────────────────────────────────────────────────────
def gaussian_innovations(sd: float = 1.0) -> Callable[[int, np.random.Generator], np.ndarray]:
    """Standard Gaussian innovation factory (the well-specified default)."""
    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, sd, n)
    return draw


def student_t_innovations(df: float = 3.0, scale: float = 1.0) \
        -> Callable[[int, np.random.Generator], np.ndarray]:
    """Heavy-tailed (Student-t) innovation factory (misspecification stress).

    ``df=3`` has finite variance but heavy tails; smaller ``df`` is heavier.
    Scaled so the innovation variance is comparable to a unit Gaussian for
    ``df > 2`` (variance = df/(df-2)); we rescale to unit variance times ``scale``.
    """
    def draw(n: int, rng: np.random.Generator) -> np.ndarray:
        raw = rng.standard_t(df, n)
        if df > 2:
            raw = raw / np.sqrt(df / (df - 2.0))  # unit variance
        return scale * raw
    return draw


# ── AR(1) / OU ───────────────────────────────────────────────────────────────
def ar1_tau_int(phi: float) -> float:
    """Analytic integrated autocorrelation time of an AR(1) with parameter ``phi``.

    ``tau_int = 1/2 * (1 + phi) / (1 - phi)`` (frames), matching the production
    M1 convention ``tau_int = 1/2 + sum_{Delta>=1} rho(Delta)``.
    """
    if not (-1.0 < phi < 1.0):
        raise ValueError("AR(1) requires -1 < phi < 1")
    return 0.5 * (1.0 + phi) / (1.0 - phi)


def ou_phi(theta_ou: float, dt: float) -> float:
    """AR(1) parameter of an OU process sampled at step ``dt``: ``exp(-theta*dt)``."""
    if theta_ou <= 0 or dt <= 0:
        raise ValueError("OU requires theta_ou > 0 and dt > 0")
    return float(np.exp(-theta_ou * dt))


def ar1_series(n: int, phi: float, rng: np.random.Generator, *,
               innovations: Optional[Callable] = None,
               burn_in: int = 200) -> np.ndarray:
    """A single stationary AR(1) series of length ``n``.

    Uses a stationary start (draws the initial value from the stationary
    distribution when innovations are Gaussian) plus a burn-in, so the returned
    series is (approximately) stationary from the first frame.
    """
    if innovations is None:
        innovations = gaussian_innovations()
    total = n + burn_in
    e = innovations(total, rng)
    x = np.empty(total, dtype=float)
    # stationary start for Gaussian: var = sigma_e^2 / (1 - phi^2); approximate by e[0]
    x[0] = e[0] / np.sqrt(max(1.0 - phi * phi, 1e-6))
    for t in range(1, total):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn_in:]


def ar2_series(n: int, a1: float, a2: float, rng: np.random.Generator, *,
               innovations: Optional[Callable] = None,
               burn_in: int = 500) -> np.ndarray:
    """A single AR(2) series (a *non-AR(1)* linear process; misspecification).

    Stability requires the roots of ``1 - a1 z - a2 z^2`` outside the unit circle;
    the caller is responsible for a stationary ``(a1, a2)``. Oscillatory choices
    (``a2 < 0``) give a shorter effective ``tau_int``; persistent choices give a
    longer one.
    """
    if innovations is None:
        innovations = gaussian_innovations()
    total = n + burn_in
    e = innovations(total, rng)
    x = np.zeros(total, dtype=float)
    for t in range(2, total):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x[burn_in:]


# ── coupled pair with a target Pearson correlation ───────────────────────────
@dataclass(frozen=True)
class SeriesPair:
    """A coupled ``(V, d)`` pair with its planted parameters (ground truth)."""

    V: np.ndarray
    d: np.ndarray
    target_r: float          # planted Pearson correlation
    phi: float               # AR(1) parameter (raw-series autocorrelation)
    tau_int_analytic: float  # analytic raw-series integrated autocorrelation time
    kind: str                # "ar1" | "ou" | "ar2" | "ar1_heavy_tailed" | ...


def coupled_ar1_pair(
    n: int, target_r: float, phi: float, rng: np.random.Generator, *,
    innovations: Optional[Callable] = None, kind: str = "ar1",
) -> SeriesPair:
    """Build a coupled ``(V, d)`` AR(1) pair with a target Pearson correlation.

    Construction: a shared latent AR(1) component ``c`` plus independent AR(1)
    idiosyncratic components ``e1, e2``, mixed as
    ``V = a c + b e1``, ``d = sign(r) a c + b e2`` with ``a = sqrt(|r|)``,
    ``b = sqrt(1 - |r|)``. Because all three latent series are AR(1)(``phi``) with
    (approximately) unit stationary variance, ``V`` and ``d`` are each AR(1)(``phi``)
    with population correlation ``target_r``. The raw-series ``tau_int`` is the
    AR(1) analytic value; the actual sample ``r`` and the product-series ``N_eff``
    are recovered/characterized through the production stack.

    ``innovations`` lets the caller inject heavy-tailed or other innovations while
    preserving the AR(1) *linear* autocorrelation (a clean misspecification knob).
    """
    if not (-1.0 <= target_r <= 1.0):
        raise ValueError("target_r must be in [-1, 1]")
    if innovations is None:
        innovations = gaussian_innovations()
    seeds = spawn_seeds(int(rng.integers(0, 2 ** 31 - 1)), 3)
    c = ar1_series(n, phi, make_rng(seeds[0]), innovations=innovations)
    e1 = ar1_series(n, phi, make_rng(seeds[1]), innovations=innovations)
    e2 = ar1_series(n, phi, make_rng(seeds[2]), innovations=innovations)
    a = np.sqrt(abs(target_r))
    b = np.sqrt(1.0 - abs(target_r))
    V = a * c + b * e1
    d = np.sign(target_r) * a * c + b * e2 if target_r != 0 else b * e2
    return SeriesPair(V=V, d=d, target_r=float(target_r), phi=float(phi),
                      tau_int_analytic=ar1_tau_int(phi), kind=kind)


def coupled_ou_pair(n: int, target_r: float, theta_ou: float, dt: float,
                    rng: np.random.Generator) -> SeriesPair:
    """Coupled OU pair (an AR(1) with ``phi = exp(-theta_ou dt)``)."""
    phi = ou_phi(theta_ou, dt)
    sp = coupled_ar1_pair(n, target_r, phi, rng, kind="ou")
    return sp


def coupled_ar2_pair(n: int, target_r: float, a1: float, a2: float,
                     rng: np.random.Generator) -> SeriesPair:
    """Coupled AR(2) pair (non-AR(1) structure; misspecification stress).

    ``tau_int_analytic`` is set to ``nan`` because the clean AR(1) formula does not
    apply; Tier B characterizes the recovered ``tau_int`` empirically rather than
    against a closed form.
    """
    if not (-1.0 <= target_r <= 1.0):
        raise ValueError("target_r must be in [-1, 1]")
    seeds = spawn_seeds(int(rng.integers(0, 2 ** 31 - 1)), 3)
    c = ar2_series(n, a1, a2, make_rng(seeds[0]))
    e1 = ar2_series(n, a1, a2, make_rng(seeds[1]))
    e2 = ar2_series(n, a1, a2, make_rng(seeds[2]))
    # standardize latent components to unit variance so the mix hits target_r
    for arr in (c, e1, e2):
        s = arr.std()
        if s > 0:
            arr /= s
    a = np.sqrt(abs(target_r))
    b = np.sqrt(1.0 - abs(target_r))
    V = a * c + b * e1
    d = np.sign(target_r) * a * c + b * e2 if target_r != 0 else b * e2
    return SeriesPair(V=V, d=d, target_r=float(target_r), phi=float("nan"),
                      tau_int_analytic=float("nan"), kind="ar2")


def coupled_heavy_tailed_pair(n: int, target_r: float, phi: float, df: float,
                              rng: np.random.Generator) -> SeriesPair:
    """Coupled AR(1) pair with heavy-tailed (Student-t) innovations."""
    return coupled_ar1_pair(n, target_r, phi, rng,
                            innovations=student_t_innovations(df=df),
                            kind="ar1_heavy_tailed")


def coupled_slow_mixing_pair(n: int, target_r: float, rng: np.random.Generator,
                             *, phi: float = 0.98) -> SeriesPair:
    """Coupled near-unit-root AR(1) pair (slow mixing; short-trajectory stress).

    Intended for *short* ``n`` where the integrated-autocorrelation-time estimate
    is known to be optimistic (under-estimates ``tau_int``); Tier B characterizes
    that under-estimation honestly.
    """
    return coupled_ar1_pair(n, target_r, phi, rng, kind="ar1_slow_mixing")
