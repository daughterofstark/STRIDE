"""M4 tests: variance-components estimation (Normal-Normal random effects with
known within-replicate variances).

Theoretical anchors
-------------------
Model: y_k ~ N(beta, tau^2 + v_k), v_k known.
* As K grows, (hat beta, hat tau^2) -> (beta, tau^2).
* When all y_k are equal, tau^2 = 0 (no heterogeneity).
* For K < 5 the Bayesian half-Normal path is used; tau^2 is honestly shrunk
  (conservative / upward at K=3), never collapsed to 0 when replicates disagree.
* K = 1 -> tau^2 unidentifiable (status k_too_small).

Tolerances follow IMPLEMENTATION_ROADMAP Part IV: varcomp recovery +/-15% at K=10.
"""
import numpy as np
import pytest

from mechanism.statistics import varcomp
from mechanism.statistics.varcomp import (
    VarCompResult, fit, dersimonian_laird, paule_mandel, bayesian_halfnormal,
)


def _draw(K, beta, tau, v0, seed):
    rng = np.random.default_rng(seed)
    v = np.full(K, v0)
    y = beta + rng.normal(0.0, tau, K) + rng.normal(0.0, np.sqrt(v))
    return y, v


def _mean_recovery(K, beta=2.0, tau=0.5, v0=0.04, nsim=400, method="auto"):
    betas, taus = [], []
    for s in range(nsim):
        y, v = _draw(K, beta, tau, v0, seed=s)
        r = fit(y, v, method=method)
        betas.append(r.beta)
        taus.append(r.tau2)
    return np.mean(betas), np.nanmean(taus)


# ── recovery / consistency ───────────────────────────────────────────────────
def test_recovery_k10_within_15pct():
    b, t = _mean_recovery(10)
    assert b == pytest.approx(2.0, rel=0.15)
    assert t == pytest.approx(0.25, rel=0.15)


def test_recovery_improves_with_k():
    # tau^2 estimation error at K=10 is no worse than at K=3, and all K are
    # within the roadmap tolerance band (Bayesian K=3 is honestly conservative).
    errs = {K: abs(_mean_recovery(K)[1] - 0.25) for K in (3, 5, 10)}
    assert errs[10] <= errs[3]
    assert errs[5] <= errs[3]
    assert errs[10] < 0.05 and errs[5] < 0.05


def test_beta_recovers_all_k():
    for K in (3, 5, 10):
        b, _ = _mean_recovery(K)
        assert b == pytest.approx(2.0, rel=0.05)


def test_k3_tau2_conservative_not_collapsed():
    # At K=3 with real heterogeneity, mean tau^2 must stay clearly positive
    # (honest/upward), i.e. the prior does not shrink it to ~0.
    _, t = _mean_recovery(3)
    assert t > 0.15


# ── tau^2 boundary behaviour ─────────────────────────────────────────────────
def test_tau2_zero_when_homogeneous():
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    v = np.full(5, 0.04)
    assert dersimonian_laird(y, v) == pytest.approx(0.0, abs=1e-9)
    assert paule_mandel(y, v) == pytest.approx(0.0, abs=1e-6)
    r = fit(y, v)  # K=5 -> PM
    assert r.tau2 == pytest.approx(0.0, abs=1e-6)


def test_tau2_positive_when_heterogeneous():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.full(5, 1e-4)  # tiny within -> spread is between-replicate
    assert dersimonian_laird(y, v) > 0.1
    assert paule_mandel(y, v) > 0.1


def test_paule_mandel_solves_generalized_q():
    y, v = _draw(8, 2.0, 0.6, 0.05, seed=3)
    t2 = paule_mandel(y, v)
    w = 1.0 / (v + t2)
    beta = (w * y).sum() / w.sum()
    q = (w * (y - beta) ** 2).sum()
    assert q == pytest.approx(len(y) - 1, abs=1e-4)


# ── edge cases / status flags ────────────────────────────────────────────────
def test_k1_undefined():
    r = fit(np.array([1.5]), np.array([0.04]))
    assert r.status == "k_too_small"
    assert np.isnan(r.tau2)
    assert r.beta == pytest.approx(1.5)


def test_k2_uses_bayesian_gate_uncertain():
    r = fit(np.array([1.0, 2.0]), np.array([0.04, 0.04]))
    assert r.method == "bayesian"
    assert r.status == "gate_uncertain"
    assert np.isfinite(r.tau2)


def test_method_dispatch_threshold():
    y10, v10 = _draw(10, 2.0, 0.5, 0.04, 1)
    y3, v3 = _draw(3, 2.0, 0.5, 0.04, 1)
    assert fit(y10, v10).method == "paule_mandel"
    assert fit(y3, v3).method == "bayesian"


def test_nonfinite_replicates_dropped():
    y = np.array([1.0, 2.0, np.nan, 1.5, 2.5])
    v = np.array([0.04, 0.04, 0.04, np.nan, 0.04])
    r = fit(y, v)
    assert r.n_replicates == 3  # two non-finite rows dropped


# ── known-variance pooling ───────────────────────────────────────────────────
def test_inverse_variance_pooling_weights():
    # with tau^2 ~ 0, beta is the inverse-variance weighted mean
    y = np.array([0.0, 10.0])
    v = np.array([1.0, 100.0])  # first obs ~100x more precise
    r = fit(y, v, method="dersimonian_laird")
    w = 1.0 / v
    expected = (w * y).sum() / w.sum()
    assert r.beta == pytest.approx(expected, rel=1e-6)
    assert r.beta < 1.0  # pulled toward the precise observation


def test_beta_se_decreases_with_k():
    y10, v10 = _draw(10, 2.0, 0.3, 0.04, 7)
    se10 = fit(y10, v10).beta_se
    se3 = fit(y10[:3], v10[:3]).beta_se
    assert se10 < se3


# ── Bayesian path properties ─────────────────────────────────────────────────
def test_bayesian_deterministic():
    y, v = _draw(3, 2.0, 0.5, 0.04, 11)
    a = bayesian_halfnormal(y, v)
    b = bayesian_halfnormal(y, v)
    assert a == b  # no RNG anywhere


def test_bayesian_prior_scale_monotone():
    # a larger half-Normal scale admits more heterogeneity -> tau^2 not smaller
    y = np.array([0.0, 1.5, 3.0])
    v = np.full(3, 0.01)
    _, _, t_small, _ = bayesian_halfnormal(y, v, prior_scale=0.05)
    _, _, t_large, _ = bayesian_halfnormal(y, v, prior_scale=5.0)
    assert t_large >= t_small


def test_bayesian_tau2_posterior_sd_finite():
    y, v = _draw(3, 2.0, 0.5, 0.04, 5)
    r = fit(y, v)
    assert np.isfinite(r.tau2_sd) and r.tau2_sd >= 0.0


def test_result_is_frozen_dataclass():
    r = fit(np.array([1.0, 2.0, 3.0]), np.full(3, 0.04))
    assert isinstance(r, VarCompResult)
    try:
        r.beta = 0.0
        raised = False
    except Exception:
        raised = True
    assert raised  # frozen
