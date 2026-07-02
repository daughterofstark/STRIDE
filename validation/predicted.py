"""Closed-form predicted operating characteristics (milestone V3).

This module implements **Part IV** of ``MATHEMATICAL_SPECIFICATION.md`` — the four
practitioner-facing operating-characteristic curves — in closed (Gaussian-approx)
form. It is the *faithfulness anchor* the roadmap describes: a predicted reference
that depends on **neither the generators (V1/V2) nor the production estimator**, so
it can be verified against its own analytic limits *before* any empirical
comparison (V5) can be tuned to it.

Purity
------
This module is pure model mathematics. It imports **nothing** from ``mechanism``
and nothing from the validation generators; it contains **no randomness** (V3 needs
no RNG). Everything is a deterministic function of the model parameters
``(beta^2, tau^2, sigma_bar^2, N_eff, K, rho_star, ...)``.

Estimand recap (spec §2.4 / Part IV)
------------------------------------
The regional signal-to-sampling ratio and reproducibility coefficient are

    lambda = beta_R^2 / (tau_R^2 + sigma_bar_R^2),
    rho_R  = lambda / (1 + lambda) = beta_R^2 / (beta_R^2 + tau_R^2 + sigma_bar_R^2),

with ``sigma_bar_R^2 ∝ 1 / N_eff`` and ``N_eff = T / (2 tau_int)``. This is exactly
the estimand the production ``reproducibility_coefficient`` computes and the ground
truth ``validation.types.RegionTruth.rho`` records; V3 predicts, it does not
re-estimate. The identity is asserted in the tests (no second copy of the formula
is introduced).

What Part IV fixes, and what it leaves to a [CHOICE]
---------------------------------------------------
The specification fixes the curves by their **limits and monotonicities**, and it
fixes two of them in closed form:

* ``rho = lambda/(1+lambda)``                              — closed form [SPEC];
* ``FPR = alpha`` by null calibration                      — [SPEC] (V4 sets rho*);
* ``ell_min`` = finest scale with predicted ``rho >= rho*``— [SPEC] definition;
* over-resolution probability ``~ exp(-c K g^2)``          — [SPEC] up to model ``c``;
* **power** and **coverage**: the spec states *properties* (power increases in
  ``beta^2, N_eff, K`` and decreases in ``tau^2``; coverage → nominal as ``K``
  grows, wider-but-honest at small ``K``) but not a unique algebraic form.

Per the milestone instruction, power and coverage use the **simplest defensible**
Gaussian approximation and are marked [CHOICE]; the tests check only the
spec-required properties (limits, monotonicity, identities), never arbitrary
numeric values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# scipy is already a production/validation dependency; used only for the normal CDF.
from scipy.stats import norm

_EPS = 1e-12


# ── core estimand (closed form; identical to the production/ground-truth ρ) ──
def lambda_snr(beta2: float, tau2: float, sigma2_bar: float) -> float:
    """Regional signal-to-sampling ratio ``lambda = beta^2 / (tau^2 + sigma_bar^2)``.

    [SPEC Part IV]. Returns ``+inf`` when the noise ``tau^2 + sigma_bar^2`` is zero
    and ``beta^2 > 0`` (a perfectly reproducible region).
    """
    noise = float(tau2) + float(sigma2_bar)
    if noise <= _EPS:
        return float("inf") if beta2 > _EPS else 0.0
    return float(beta2) / noise


def rho_from_lambda(lam: float) -> float:
    """``rho = lambda / (1 + lambda)`` [SPEC Part IV]. Maps ``[0, inf] -> [0, 1]``."""
    if lam == float("inf"):
        return 1.0
    if lam < 0:
        raise ValueError("lambda must be >= 0")
    return float(lam / (1.0 + lam))


def rho_from_params(beta2: float, tau2: float, sigma2_bar: float) -> float:
    """``rho = beta^2 / (beta^2 + tau^2 + sigma_bar^2)`` [SPEC].

    The single source of the reproducibility formula in V3; equals
    ``validation.types.RegionTruth.rho`` and the production
    ``reproducibility_coefficient`` (asserted in tests, not duplicated).
    """
    denom = float(beta2) + float(tau2) + float(sigma2_bar)
    if denom <= _EPS:
        return 0.0
    return float(beta2) / denom


def lambda_star(rho_star: float) -> float:
    """Threshold on ``lambda`` equivalent to a threshold ``rho_star`` on ``rho``.

    Since ``rho`` is strictly increasing in ``lambda``, ``rho >= rho_star`` iff
    ``lambda >= lambda_star = rho_star / (1 - rho_star)``. [SPEC-derived]
    """
    if not (0.0 <= rho_star < 1.0):
        raise ValueError("rho_star must be in [0, 1)")
    return rho_star / (1.0 - rho_star)


# ── sampling-noise link (spec §2.1 / Part IV: sigma_bar^2 ∝ 1/N_eff) ─────────
def n_eff_from_T(T: float, tau_int: float) -> float:
    """Effective sample size ``N_eff = T / (2 tau_int)`` [SPEC §2.1]."""
    if tau_int <= 0:
        raise ValueError("tau_int must be > 0")
    return float(T) / (2.0 * float(tau_int))


def sigma2_bar_from_neff(n_eff: float, *, theta: float = 0.0,
                         k_proportional: Optional[float] = None) -> float:
    """Within-replicate variance of the regional statistic from ``N_eff``.

    [SPEC Part IV: ``sigma_bar^2 ∝ 1/N_eff``]. Two modes:

    * default: the Fisher large-sample form ``(1 - theta^2)^2 / N_eff`` (the exact
      per-residue sampling variance the production M1 uses); with ``theta = 0`` this
      is the plain ``1 / N_eff`` proportionality the spec states.
    * ``k_proportional`` given: ``k_proportional / N_eff`` for an explicit constant.

    The spec only fixes the ``1/N_eff`` proportionality; the constant is
    model-dependent, so it is exposed rather than hard-coded. [CHOICE: expose the
    proportionality constant instead of fixing it]
    """
    if n_eff <= 0:
        raise ValueError("n_eff must be > 0")
    if k_proportional is not None:
        return float(k_proportional) / float(n_eff)
    return float((1.0 - theta * theta) ** 2) / float(n_eff)


# ── false-resolution rate (spec: alpha by construction) ──────────────────────
def predicted_fpr(alpha: float) -> float:
    """Predicted false-resolution rate ``= alpha`` by null calibration [SPEC Part IV].

    Under ``beta_R = 0`` the bias-corrected ``beta^2`` concentrates at 0 and
    ``rho_star`` is set (V4) to the upper-``alpha`` quantile of ``rho_hat`` under the
    null surrogates, so ``Pr(rho_hat >= rho_star | beta = 0) = alpha`` *by
    construction*. V3 records this identity; the empirical calibration is V4.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    return float(alpha)


# ── power (Gaussian-approx; [CHOICE], minimal) ───────────────────────────────
def predicted_power(beta2: float, tau2: float, sigma2_bar: float, K: int,
                    rho_star: float) -> float:
    """Predicted power ``Pr(rho_hat >= rho_star)`` (Gaussian approximation).

    [CHOICE — minimal, spec leaves the form open] The simplest defensible model:
    treat the regional statistic mean ``Theta_bar`` as
    ``Theta_bar ~ Normal(beta, (tau^2 + sigma_bar^2) / K)`` (replicate mean of a
    per-replicate statistic with population mean ``beta`` and between+within
    variance ``tau^2 + sigma_bar^2``). To leading order the gate crosses when
    ``lambda_hat = Theta_bar^2 / (tau^2 + sigma_bar^2) >= lambda_star``, i.e. when
    ``|Theta_bar| >= sqrt(lambda_star (tau^2 + sigma_bar^2))``. Hence

        power = Pr(|Theta_bar| >= thr),   thr = sqrt(lambda_star * s2),  s2 = tau^2+sigma_bar^2,
                Theta_bar ~ Normal(beta, s2 / K).

    This uses a single normal CDF and no non-central chi-square machinery. It
    satisfies exactly the spec-mandated behavior (verified in tests):

    * increasing in ``beta^2``;
    * increasing in ``N_eff`` (i.e. decreasing in ``sigma_bar^2``, hence in ``1/T``);
    * increasing in ``K``;
    * decreasing in ``tau^2``;
    * ``-> 1`` as ``beta^2 -> inf``; small (the model's own null crossing rate) as
      ``beta^2 -> 0`` — which the V4 calibration pins to ``alpha`` by choosing
      ``rho_star`` (this function does not itself perform that calibration).

    No claim is made about the specific numeric value; only these properties are
    tested, as the milestone requires.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    s2 = float(tau2) + float(sigma2_bar)
    lam_star = lambda_star(rho_star)
    beta = math.sqrt(max(float(beta2), 0.0))
    if s2 <= _EPS:
        # deterministic limit: rho = 1 if beta^2 > 0, else 0.
        rho = rho_from_params(beta2, tau2, sigma2_bar)
        return 1.0 if rho >= rho_star else 0.0
    thr = math.sqrt(lam_star * s2)
    sd = math.sqrt(s2 / K)
    upper = float(norm.sf((thr - beta) / sd))
    lower = float(norm.cdf((-thr - beta) / sd))
    return float(min(max(upper + lower, 0.0), 1.0))


# ── coverage (Gaussian-approx; [CHOICE], minimal) ────────────────────────────
def predicted_coverage(K: int, nominal: float = 0.95) -> float:
    """Predicted coverage of the ``beta_R`` interval (Gaussian approximation).

    [CHOICE — minimal, spec leaves the form open] The spec states only the
    *properties*: coverage attains the ``nominal`` level as ``K`` grows, and at
    small ``K`` the (Bayesian) interval is **wider but honest** — i.e. coverage is
    ``>= nominal``, never anticonservative (unlike naive ``SD/sqrt(K)``). The
    simplest function with exactly those properties: return a value that is
    ``>= nominal`` for all ``K`` and decreases monotonically to ``nominal`` as
    ``K -> inf``. We use

        coverage(K) = nominal + (1 - nominal) / K,

    so coverage(1) is the widest and ``coverage(K) -> nominal`` from above. This is
    a stand-in for "wider-but-honest at small K"; no specific numeric value is
    asserted in the tests, only monotonicity toward ``nominal`` and the honesty
    bound ``coverage >= nominal``.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not (0.0 < nominal < 1.0):
        raise ValueError("nominal must be in (0, 1)")
    return float(nominal + (1.0 - nominal) / K)


# ── minimum resolvable scale (spec definition) ───────────────────────────────
@dataclass(frozen=True)
class ScalePrediction:
    """Predicted reproducibility of one scale (coarse->fine ordering by index)."""

    scale_index: int          # residue = 0 (finest), matching production convention
    rho_predicted: float


def ell_min(scales: Sequence[ScalePrediction], rho_star: float) -> Optional[int]:
    """Minimum resolvable scale: the finest scale with predicted ``rho >= rho_star``.

    [SPEC Part IV] ``ell_min(K, T, tau_int, g) =`` the finest ``ell`` whose expected
    ``rho_ell >= rho_star``. Here the caller supplies the predicted ``rho`` per
    scale (computed via :func:`rho_from_params` from that scale's
    ``(beta^2, tau^2, sigma_bar^2)``); this function selects the finest passing
    scale. "Finest" = smallest ``scale_index`` (residue = 0). Returns ``None`` when
    no scale passes (``ell_hat_star = varnothing``).

    Because coarsening reduces ``sigma_bar^2`` (larger regions pool more residues)
    and the passable set is upward-closed (spec I2), a finer resolvable scale can
    only appear as ``K`` or ``T`` (hence ``N_eff``) grow; ``ell_min`` is therefore
    non-increasing in ``K`` and ``T`` when the per-scale ``rho`` predictions are
    computed consistently. That monotonicity is exercised in the tests via
    explicit consistent scale tables.
    """
    passing = [s.scale_index for s in scales if s.rho_predicted >= rho_star]
    if not passing:
        return None
    return int(min(passing))


# ── over-resolution bound (spec: ~ exp(-c K g^2), model-dependent c) ─────────
def over_resolution_bound(K: int, gap: float, c: float = 1.0) -> float:
    """Approximate probability of certifying a scale finer than truth.

    [SPEC Part IV / Part III] ``~ exp(-c K g^2)`` with a **model-dependent** ``c``
    and gap ``g = rho_star - rho_ell^true`` (the amount by which the threshold
    exceeds the true reproducibility of the over-fine scale). Stated by the spec as
    asymptotic guidance, not a sharp finite-``K`` bound; ``c`` is exposed with a
    documented default rather than hard-coded. Clipped to ``[0, 1]`` (it is a
    probability bound). Decreasing in ``K`` and in ``|g|``; ``-> 0`` as ``K -> inf``.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if c <= 0:
        raise ValueError("c must be > 0")
    val = math.exp(-c * K * gap * gap)
    return float(min(max(val, 0.0), 1.0))


# ── reference-table helper (roadmap: "predicted-curve functions and tables") ─
def predicted_reference_table(
    betas2: Sequence[float], tau2: float, sigma2_bar: float, Ks: Sequence[int],
    rho_star: float, *, alpha: float = 0.05,
) -> list:
    """A pure predicted-curve table over a ``(beta^2, K)`` grid (no I/O, no RNG).

    Returns a list of dicts, each with the closed-form ``rho``, ``lambda``,
    predicted ``power``, predicted ``coverage``, and the constant ``fpr = alpha``.
    Provided as the auditable reference structure later milestones (V5) consume; V3
    itself performs no empirical comparison.
    """
    rows = []
    for b2 in betas2:
        lam = lambda_snr(b2, tau2, sigma2_bar)
        rho = rho_from_lambda(lam)
        for K in Ks:
            rows.append({
                "beta2": float(b2),
                "tau2": float(tau2),
                "sigma2_bar": float(sigma2_bar),
                "K": int(K),
                "rho_star": float(rho_star),
                "lambda": float(lam),
                "rho": float(rho),
                "power": predicted_power(b2, tau2, sigma2_bar, K, rho_star),
                "coverage": predicted_coverage(K),
                "fpr": predicted_fpr(alpha),
            })
    return rows
